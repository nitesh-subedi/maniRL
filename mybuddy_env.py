import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
from scipy.spatial import distance
import time
from collections import deque
import gc
import torch



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
        self.world.initialise(usd_path="omniverse://localhost/Users/nitesh/plant_v21/plant_v21.usd", 
                              env_usd = "omniverse://localhost/Users/nitesh/env_v2/environment.usd",
                              hdr_path="omniverse://localhost/Users/nitesh/env_v2/textures/rosendal_plains_2_4k.hdr")

        from mybuddy.robot import Robot
        self.robot = Robot(
            urdf_path="/home/nitesh/workspace/rosws/mybuddy_robot_rl/src/mybuddy_description/urdf/urdf.urdf",
            world=self.world.world, simulation_app=self._simulation_app)

        import omni
        omni.timeline.get_timeline_interface().play()
        # self.robot.initialise_control_interface()

        self.world._world.step()
        gc.collect()
        self._simulation_app.update()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(256, 256, 3),
            dtype=np.uint8
        )
        self.goal_space = spaces.Box(
            low=0,
            high=255,
            shape=(256, 256, 3),
            dtype=np.uint8
        )

        self.observation_space = spaces.Dict({
            'observation': self.observation_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space
        })

        self.last_angles = np.zeros(6)

        self.lower_red = np.array([90, 50, 50])
        self.upper_red = np.array([130, 255, 255])

        self.lower_green = np.array([35, 40, 40])
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
        self.previous_magnitude_spectrum = None
        self.orb = cv2.ORB_create()
        obs = self.get_observation()
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)

        # Create masks for the red cube and green plant
        plant_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Count non-zero pixels for cube and plant
        self.init_green_pixels = cv2.countNonZero(plant_mask)
        action = [-90, 30, 130,-120, 0, 0]
        self.robot.send_angles(0, action, degrees=True)
        for i in range(100):
            self.world._world.step()
        self.goal_image = self.get_observation()
        # save image
        cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_goal.jpg", self.goal_image)


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
        self.world._world.step()
        gc.collect()
        collision, end_effector_collision = self.robot.check_collision()
        self.last_angles = action
        obs = self.get_observation()
        reward = self.compute_reward(obs, self.goal_image, {})
        # intrinsic_reward = self.get_intrinsic_reward(action)
        # exploration_bonus = self.get_exploration_bonus(obs)

        # Combine rewards
        total_reward = reward #+ intrinsic_reward * 0.3 + exploration_bonus * 0.1
        # if end_effector_collision:
        #     total_reward -= 20

        done = self.get_done(collision)
        self.episode_length += 1
        truncated = self.is_truncated()

        return {
            'observation': obs,
            'desired_goal': self.goal_image,
            'achieved_goal': obs.copy()
        }, float(total_reward), done, truncated, {}

    @staticmethod
    def get_done(collision):
        if collision:
            return True
        return False

    def get_observation(self):
        image = self.world.get_image()
        image = cv2.resize(image, (256, 256))

        return image

    def get_reward(self, obs, collision, action):
        
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)

        # Crop image to focus on the cube, the cube is located at the center of the image
        cube_hsv = hsv.copy()[90:140, 100:150]

        # Create masks for the red cube and green plant
        cube_mask = cv2.inRange(cube_hsv, self.lower_red, self.upper_red)
        # plant_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # if time.time() - self.tic > 10:
        #     self.tic = time.time()
        #     cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_cube.jpg", cv2.bitwise_and(obs[90:140, 100:150], obs[90:140, 100:150], mask=cube_mask))
        #     cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_plant.jpg", plant_mask)
        
        # Count non-zero pixels for cube and plant
        cube_pixels = cv2.countNonZero(cube_mask)
        if cube_pixels > 2000:
            reward = 1
        else:
            reward = 0
        # green_pixels = cv2.countNonZero(plant_mask)
        # reward = ((green_pixels - self.init_green_pixels) / self.init_green_pixels) * 100
        # if green_pixels < self.init_green_pixels:
        #     self.init_green_pixels = green_pixels
        # Normalize cube pixel count
        # normalized_cube_pixels = max(cube_pixels, 0) / 2396
        
        # if time.time() - self.tic > 5:
        #     self.tic = time.time()
        #     print(f"Reward: {reward}")

        # # Reward/punishment based on cube visibility
        # if cube_pixels <= 400:
        #     reward -= 5
        #     self.observing_cube = False
        # else:
        #     self.observing_cube = True
        #     if self.last_pixels < cube_pixels:
        #         reward += 150
        #         self.last_good_actions = action
        #         self.last_pixels = cube_pixels

        # Penalize collision
        # if collision:
        #     reward -= 50

        # Intrinsic reward decay
        # if not np.allclose(self.last_good_actions[:5], self.initial_angles[:5]) and not self.observing_cube:
        #     stagnation_penalty = np.exp(-np.linalg.norm(action - self.last_good_actions)) * 10
        #     reward -= stagnation_penalty

        return reward


    def is_truncated(self):
        if self.world._world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            return True
        # if self.episode_length >= self._max_episode_length:
        #     return True
        return False

    def reset(self, seed=None):
        self.world._world.stop()
        self.world._world.reset()
        torch.cuda.empty_cache()
        self.robot.send_angles(0, self.initial_angles, degrees=False)
        torch.cuda.empty_cache()
        for i in range(30):
            self.world._world.step()
        gc.collect()
        self.world.goal_cube.set_world_pose([np.random.uniform(-0.02, 0.02), -0.4, 0.22], [0, 0, 0, 1])
        self.episode_length = 0
        self.last_angles = self.initial_angles
        # self.previous_actions = []
        self.w = {}
        self.observing_cube = False
        self.total_visits = 0
        return {
            'observation': self.get_observation(),
            'desired_goal': self.goal_image,
            'achieved_goal': self.get_observation().copy()}, {}
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward based on the achieved_goal and desired_goal.
        Handle both single and batch computation by checking the input shape.
        """
        # Convert to float to prevent overflow
        achieved_goal = achieved_goal.astype(np.float32)
        desired_goal = desired_goal.astype(np.float32)

        if achieved_goal.ndim == 3:  # Single goal (no batch dimension)
            distance = np.sum(np.abs(achieved_goal - desired_goal))
            if distance == 0:
                reward = 1  # Reward for achieving the goal
            else:
                reward = -distance / (3 * 256 * 256 * 255)  # Negative reward based on distance

        else:  # Batch computation
            distance = np.sum(np.abs(achieved_goal - desired_goal), axis=(1, 2, 3))
            reward = np.where(distance == 0, 1, -distance / (3 * 256 * 256 * 255))
        if time.time() - self.tic > 5:
            self.tic = time.time()
            print(f"Reward: {reward}")
        return reward

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
        if closest_abstraction is not None and min_hamming_distance < self.hamming_threshold():
            self.w[closest_abstraction] += 1
            pseudo_count = self.w.get(state_abstraction, 0)
        else:
            self.w[state_abstraction] = 1
            pseudo_count = 1

        smoothed_bonus = np.sqrt(np.log(1 + self.total_visits) / (1 + (0.9 * pseudo_count + 0.1 * pseudo_count)))
        self.total_visits += 1

        # if time.time() - self.tic > 5:
        #     self.tic = time.time()
        #     print(f"Exploration Bonus: {smoothed_bonus}")
        return smoothed_bonus

    def state_abstraction(self, obs):
        # Use ORB (Oriented FAST and Rotated BRIEF) for better feature extraction from images
        keypoints, descriptors = self.orb.detectAndCompute(obs, None)
        hash_value = np.sum(descriptors) if descriptors is not None else 0
        state_abstraction = hash_value % 10000  # Hash to a manageable size

        return state_abstraction

    def hamming_threshold(self):
        # Define a threshold for the Hamming distance to consider states as similar
        return 10
