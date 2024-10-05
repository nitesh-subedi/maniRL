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
        
        # Attach ee camera to the robot
        self.world.add_ee_camera()

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
            shape=(256, 256, 6),
            dtype=np.uint8
        )

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
        self.world._world.render()
        self.world._world.render()
        self.world._world.render()
        cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_ee.jpg", self.world.get_ee_image())

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
        reward, self.cube_pixels = self.get_reward(obs, collision, action)
        intrinsic_reward = self.get_intrinsic_reward(action)
        # intrinsic_reward = 0
        # if not self.observing_cube:
        #     intrinsic_reward = self.get_intrinsic_reward(action)
        # exploration_bonus = self.get_exploration_bonus(obs)
        # if time.time() - self.tic > 5:
        #     self.tic = time.time()
        #     print(f"Exploration Bonus: {exploration_bonus}")

        # Combine rewards
        total_reward = reward * 0.6 + intrinsic_reward * 0.3 #+ exploration_bonus * 0.1
        if end_effector_collision:
            total_reward -= 20

        done = self.get_done(collision, self.cube_pixels)
        self.episode_length += 1
        truncated = self.is_truncated()

        return np.concatenate([obs[0], obs[1]], axis=-1), float(total_reward), done, truncated, {}

    @staticmethod
    def get_done(collision, cube_pixels):
        if collision:
            return True
        # if cube_pixels > 2000:
        #     return True
        return False

    def get_observation(self):
        image = self.world.get_image()
        image = cv2.resize(image, (256, 256))
        ee_image = self.world.get_ee_image()
        ee_image = cv2.resize(ee_image, (256, 256))
        if time.time() - self.tic > 10:
            self.tic = time.time()
            cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image.jpg", image)
            cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_ee.jpg", ee_image)
        # Stack images with shape (2, 256, 256, 3)
        return np.stack([image, ee_image], axis=0)

    def get_reward(self, obs, collision, action):
        # Initialize variables for cube and plant pixel counts across both images
        total_cube_pixels = 0
        total_green_pixels = 0
        reward = 0

        # Process each image in the observation (obs[0] and obs[1])
        for i in range(2):
            # Convert the image to HSV color space
            hsv = cv2.cvtColor(obs[i], cv2.COLOR_BGR2HSV)
            if i == 1:
                cube_hsv = hsv.copy()
            # Crop image to focus on the cube (assuming the cube is at the center in both images)
            else:
                cube_hsv = hsv.copy()[90:140, 100:150]
            
            # Create masks for the red cube and green plant
            cube_mask = cv2.inRange(cube_hsv, self.lower_red, self.upper_red)
            plant_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            # Count non-zero pixels for the cube and plant
            cube_pixels = cv2.countNonZero(cube_mask)
            green_pixels = cv2.countNonZero(plant_mask)
            
            # Accumulate pixel counts across both images
            total_cube_pixels += cube_pixels
            total_green_pixels += green_pixels

            # Normalize the cube pixel count (assuming the same maximum pixel count for both images)
            normalized_cube_pixels = max(cube_pixels, 0) / 2396
            reward += normalized_cube_pixels * 100 - green_pixels / 1000

            # Reward/punishment based on cube visibility
            if cube_pixels <= 400:
                reward -= 5
                self.observing_cube = False
            else:
                self.observing_cube = True
                if self.last_pixels < total_cube_pixels:
                    reward += 150
                    self.last_good_actions = action
                    self.last_pixels = total_cube_pixels

        # Penalize collision (applied once for both images)
        if collision:
            reward -= 20

        # Intrinsic reward decay (applied once for both images)
        if not np.allclose(self.last_good_actions[:5], self.initial_angles[:5]) and not self.observing_cube:
            stagnation_penalty = np.exp(-np.linalg.norm(action - self.last_good_actions)) * 10
            reward -= stagnation_penalty

        # Return reward and total cube pixel count across both images
        return reward + 0.1, total_cube_pixels



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
        obs = self.get_observation()
        return np.concatenate([obs[0], obs[1]], axis=-1), {}

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
