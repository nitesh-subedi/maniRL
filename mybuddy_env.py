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
        self.world.initialise(usd_path="omniverse://localhost/Users/nitesh/plant_v19/plant_v19.usd")

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
        self.last_angles = np.zeros(6)
        # Define the range for green color
        self.lower_red1 = np.array([0, 120, 70])    # Lower bound for the first red range
        self.upper_red1 = np.array([10, 255, 255])  # Upper bound for the first red range

        self.lower_red2 = np.array([170, 120, 70])  # Lower bound for the second red range
        self.upper_red2 = np.array([180, 255, 255]) # Upper bound for the second red range

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
        exploration_bonus = self.get_exploration_bonus(obs)
        # if time.time() - self.tic > 5:
        #     self.tic = time.time()
        #     print(f"Exploration Bonus: {exploration_bonus}")

        # Combine rewards
        total_reward = reward * 0.6 + intrinsic_reward * 0.3 + exploration_bonus * 0.1
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
        image = self.world.get_image()
        image = cv2.resize(image, (256, 256))

        return image

    def get_reward(self, obs, collision, action):
        camera_frame = obs
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

        # Create a mask for red color
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2

        cube_pixels = cv2.countNonZero(mask)
        normalized_pixels = (np.maximum(cube_pixels - 500, 0)) / (2396 - 500)
        reward = normalized_pixels * 100

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            if contour_area > 500:  # Minimum area threshold
                reward += contour_area * 0.05  # Small reward for seeing the cube clearly

        # # Step 3: Extract the green channel (from the original image, not HSV)
        # green_channel = camera_frame[:, :, 1]  # Green channel in BGR

        # # Step 4: Apply FFT on the green channel
        # fft_green = np.fft.fft2(green_channel)

        # # Step 5: Shift the zero frequency component to the center
        # fft_green_shifted = np.fft.fftshift(fft_green)

        # # Step 6: Compute the magnitude spectrum
        # magnitude_spectrum = np.abs(fft_green_shifted)
        # if self.previous_magnitude_spectrum is not None:
        #     reward = -np.linalg.norm(magnitude_spectrum - self.previous_magnitude_spectrum)


        if cube_pixels <= 400:
            reward -= 5
            self.observing_cube = False
        else:
            self.observing_cube = True
            if self.last_pixels < cube_pixels:
                reward += 150
                self.last_good_actions = action
                self.last_pixels = cube_pixels
        

        if collision:
            reward -= 20

        if not (self.last_good_actions[:5] == self.initial_angles[:5]).any() and not self.observing_cube:
            # Decay the intrinsic reward if the action hasn't changed much over time
            stagnation_penalty = np.exp(-np.linalg.norm(action - self.last_good_actions)) * 10
            reward -= stagnation_penalty

        
        reward += 1
        # self.previous_magnitude_spectrum = magnitude_spectrum

        return reward, cube_pixels

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
        if closest_abstraction is not None and min_hamming_distance < self.hamming_threshold():
            self.w[closest_abstraction] += 1
            pseudo_count = self.w.get(state_abstraction, 0)
        else:
            self.w[state_abstraction] = 1
            pseudo_count = 1

        smoothed_bonus = np.sqrt(np.log(1 + self.total_visits) / (1 + (0.9 * pseudo_count + 0.1 * pseudo_count)))
        self.total_visits += 1

        if time.time() - self.tic > 5:
            self.tic = time.time()
            print(f"Exploration Bonus: {smoothed_bonus}")
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
