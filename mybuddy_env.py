import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
from scipy.spatial import distance
import time
from collections import deque
from PIL import Image
import imagehash
from omni.isaac.lab.sim.converters import MeshConverter



class MyBuddyEnv(gym.Env):
    def __init__(self,
                 skip_frame=1,
                 physics_dt=1.0 / 60.0,
                 rendering_dt=1.0 / 60.0,
                 max_episode_length=1000,
                 seed=0,
                 config={"headless": True, "anti_aliasing": 1}) -> None:
        super().__init__()
        self._simulation_app = SimulationApp(launch_config=config)
        self._simulation_app.set_setting("/app/window/drawMouse", True)
        self._simulation_app.set_setting("/app/livestream/proto", "ws")
        self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        self._simulation_app.set_setting("/ngx/enabled", False)

        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.services.streamclient.webrtc")

        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        # Import world
        from world.world import SimulationEnv
        self.world = SimulationEnv(config={"physics_dt": physics_dt, "rendering_dt": rendering_dt})
        self.world.initialise()

        from mybuddy.robot import Robot
        self.robot = Robot(
            urdf_path="/home/nitesh/workspace/rosws/mybuddy_robot_rl/src/mybuddy_description/urdf/urdf.urdf",
            world=self.world.world, simulation_app=self._simulation_app)

        import omni
        omni.timeline.get_timeline_interface().play()
        # self.robot.initialise_control_interface()

        self.world._world.step()
        self._simulation_app.update()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(256, 256, 3),
            dtype=np.uint8
        )
        self.last_angles = np.zeros(6)
        # Define the range for green color
        self.lower_green = np.array([35, 100, 100])
        self.upper_green = np.array([85, 255, 255])
        self.episode_length = 0
        self.initial_angles = np.deg2rad([0, -110, 120, -120, -95, 0])
        # Initialize parameters for EXPLORS
        self.w = {}  # Initialize state visitation counts for exploration bonus

        # Initialize list to store previous actions for intrinsic reward calculation
        max_length = 1000
        self.previous_actions = deque(maxlen=max_length)
        self.last_good_actions = self.initial_angles
        self.last_pixels = 0
        self.observing_cube = False
        self.tic = time.time()
        self.total_visits = 0
        self.cube_pixels = 0

    def step(self, action):
        # if time.time() - self.tic > 3:
        #     self.world.goal_cube.set_world_pose([np.random.uniform(-0.06, 0.06), -0.4, 0.22], [0, 0, 0, 1])
        #     self.tic = time.time()
        action = np.clip(action, -1.0, 1.0)
        # if self.cube_pixels > 2000:
        #     action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # else:
        if self.observing_cube:
            action = np.array([action[0], action[1], action[2], action[3], action[4], 0.0]) * 0.05
        else:
            action = np.array([action[0], action[1], action[2], action[3], action[4], 0.0]) * 0.1
        action = self.last_angles + action
        self.robot.send_angles(0, action, degrees=False)
        for i in range(2):
            self.world._world.step()
        collision, end_effector_collision = self.robot.check_collision()
        self.last_angles = action
        obs = self.get_observation()
        reward, self.cube_pixels = self.get_reward(obs, collision, action)
        intrinsic_reward = self.get_intrinsic_reward(action)
        # intrinsic_reward = 0
        # if not self.observing_cube:
        #     intrinsic_reward = self.get_intrinsic_reward(action)
        exploration_bonus = self.get_exploration_bonus(obs)
        # if time.time() - self.tic > 5:
        #     self.tic = time.time()
        #     print(f"Exploration Bonus: {exploration_bonus}")

        # Combine rewards
        total_reward = reward + intrinsic_reward + exploration_bonus
        if end_effector_collision:
            total_reward -= 20

        done = self.get_done(collision, self.cube_pixels)
        self.episode_length += 1
        truncated = self.is_truncated()

        return obs, float(total_reward), done, truncated, {}

    @staticmethod
    def get_done(collision, cube_pixels):
        if collision:
            return True
        # if cube_pixels > 2000:
        #     return True
        return False

    def get_observation(self):
        self.world._world.render()
        self.world._world.step()
        image = self.world.get_image()
        image = cv2.resize(image, (256, 256))

        return image

    def get_reward(self, obs, collision, action):
        camera_frame = obs
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

        # Create a mask for green color
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        cube_pixels = cv2.countNonZero(mask)
        normalized_pixels = cube_pixels / 2396.0
        reward = normalized_pixels * 150

        if cube_pixels == 0:
            reward -= 5
            self.observing_cube = False
        else:
            self.observing_cube = True
            if self.last_pixels < cube_pixels:
                reward += 50
                self.last_good_actions = action
                self.last_pixels = cube_pixels
            # now = datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            # cv2.imwrite(f"/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image{current_time}.png",
            #             camera_frame)
        

        if collision:
            reward -= 20

        if not (self.last_good_actions[:5] == self.initial_angles[:5]).any() and not self.observing_cube:
            reward -= np.abs(np.linalg.norm(action - self.last_good_actions)) * 10
        
        # if time.time() - self.tic > 5:
        #     self.tic = time.time()
        #     print(f"Good Actions: {self.last_good_actions}")
        #     print(f"Angle error: {np.abs(np.linalg.norm(action - self.last_good_actions))}")
        
        reward += 1

        return reward, cube_pixels

    def is_truncated(self):
        if self.world._world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            return True
        # if self.episode_length >= self._max_episode_length:
        #     return True
        return False

    def reset(self, seed=None):
        self.world._world.reset()
        self.world._world.reset()
        self.robot.send_angles(0, self.initial_angles, degrees=False)
        for i in range(30):
            self.world._world.step()
        self.world.goal_cube.set_world_pose([np.random.uniform(-0.02, 0.02), -0.4, 0.22], [0, 0, 0, 1])
        self.episode_length = 0
        self.last_angles = self.initial_angles
        # self.previous_actions = []
        self.w = {}
        self.observing_cube = False
        self.total_visits = 0
        return self.get_observation(), {}

    def render(self, mode="human"):
        pass

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def get_intrinsic_reward(self, action):
        # Compute intrinsic reward based on action novelty
        decay_factor = 0.9

        if not self.previous_actions:
            intrinsic_reward = 0.1
        else:
            weighted_distances = []
            for i, prev_action in enumerate(self.previous_actions):
                weight = decay_factor ** (len(self.previous_actions) - i - 1)
                distance_to_prev_action = distance.euclidean(action, prev_action)
                weighted_distances.append(weight * distance_to_prev_action)

            min_weighted_distance = min(weighted_distances)
            intrinsic_reward = 1.0 / (1.0 + min_weighted_distance) * 0.1

        self.previous_actions.append(action)
        return intrinsic_reward

    def get_exploration_bonus(self, obs):
        state_abstraction = self.state_abstraction(obs)

        closest_abstraction = None
        min_hamming_distance = float('inf')

        # Find the closest abstraction in terms of Hamming distance
        for abstraction in self.w.keys():
            hamming_distance = state_abstraction - abstraction
            if hamming_distance < min_hamming_distance:
                min_hamming_distance = hamming_distance
                closest_abstraction = abstraction

        # Update the visit count based on the closest state
        if closest_abstraction is not None and min_hamming_distance < self.hamming_threshold:
            self.w[closest_abstraction] += 1
            pseudo_count = self.w[closest_abstraction]
        else:
            self.w[state_abstraction] = 1
            pseudo_count = 1

        bonus = np.sqrt(np.log(1 + self.total_visits) / (1 + pseudo_count))
        self.total_visits += 1

        if time.time() - self.tic > 5:
            self.tic = time.time()
            print(f"Exploration Bonus: {bonus}")
        return bonus * 5

    def state_abstraction(self, obs):
        # Convert the observation into an image format if necessary
        obs_image = Image.fromarray(obs)
        # Use average hashing to create a hash of the observation
        return imagehash.average_hash(obs_image)

    @property
    def hamming_threshold(self):
        # Define a threshold for the Hamming distance to consider states as similar
        return 10
