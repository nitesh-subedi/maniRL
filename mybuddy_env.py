import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
from scipy.spatial import distance
import time
from collections import deque
from PIL import Image
import sys
import gc
import torch
from mybuddy.utils import DepthEstimator



class MyBuddyEnv(gym.Env):
    def __init__(self,
                 skip_frame=1,
                 physics_dt=1.0 / 60.0,
                 rendering_dt=1.0 / 60.0,
                 max_episode_length=1000,
                 seed=0,
                 config={"headless": True, "anti_aliasing": 1}) -> None:
        super().__init__()
        self.depth_estimator = DepthEstimator()
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
        image_obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(256, 256, 3),
            dtype=np.uint8
        )

        # Define the end-effector position space (3 values: x, y, z)
        end_effector_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0]),
            high=np.array([2.0, 2.0, 2.0]),
            dtype=np.float64
        )

        # Combine both spaces using Tuple or Dict (depending on your needs)
        self.observation_space = spaces.Dict({
            'image': image_obs_space,
            'end_effector_pos': end_effector_space
        })
        self.last_angles = np.zeros(6)

        self.lower_red = np.array([90, 50, 50])
        self.upper_red = np.array([130, 255, 255])

        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])

        self.episode_length = 0
        self.initial_angles = np.deg2rad([0, -110, 120, -120, -95, 0])
        # # Initialize parameters for EXPLORS
        # self.w = {}  # Initialize state visitation counts for exploration bonus

        # # Initialize list to store previous actions for intrinsic reward calculation
        max_length = 1000
        self.previous_actions = deque(maxlen=max_length)
        # self.last_good_actions = self.initial_angles
        # self.last_pixels = 0
        # self.observing_cube = False
        self.tic = time.time()
        # self.total_visits = 0
        # self.previous_magnitude_spectrum = None
        

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
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
        ee_location = np.array(self.robot.get_ee_position())
        # print(f"EE Location: {ee_location}")
        reward = self.get_reward(obs, collision, ee_location)
        int_reward = self.get_intrinsic_reward(action)
        # Combine rewards
        total_reward = reward + int_reward
        if end_effector_collision:
            total_reward -= 2

        done = self.get_done(collision)
        self.episode_length += 1
        truncated = self.is_truncated()
        observation = {
            'image': obs,
            'end_effector_pos': ee_location
        }
        return observation, float(total_reward), done, truncated, {}

    @staticmethod
    def get_done(collision):
        if collision:
            return True
        return False

    def get_observation(self):
        image = self.world.get_image()
        image = cv2.resize(image, (256, 256))

        return image

    def get_reward(self, obs, collision, ee_location):
        # Compute reward based on the observation
        unwanted_pixels, result = self.depth_estimator.remove_nearest_objects(obs)
        if time.time() - self.tic > 5:
            self.tic = time.time()
            cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_goal_output.jpg", result)
        # More reward if less unwanted pixels
        reward = - (unwanted_pixels / 60000) * 10

        # goal_location = np.array([0, -0.6])
        # # More reward if closer to goal
        # reward -= np.linalg.norm(ee_location[:2] - goal_location) * 10

        # Penalize collision
        if collision:
            reward -= 20

        return reward + 0.1


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
        ee_location = np.array(self.robot.get_ee_position())
        observation = {
            'image': self.get_observation(),
            'end_effector_pos': ee_location
        }
        return observation, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def get_intrinsic_reward(self, action):
        decay_factor = 0.99

        if not self.previous_actions:
            intrinsic_reward = 10.0
        else:
            prev_actions = np.array(self.previous_actions)
            
            # Compute Euclidean distances between the current action and all previous actions
            distances = np.linalg.norm(prev_actions - action, axis=1)
            
            # Generate decay weights for all previous actions
            weights = decay_factor ** np.arange(len(self.previous_actions) - 1, -1, -1)
            
            # Compute weighted distances
            weighted_distances = weights * distances
            
            # Find the minimum weighted distance
            min_weighted_distance = np.min(weighted_distances)
            
            # Compute intrinsic reward
            intrinsic_reward = 1.0 / (1.0 + min_weighted_distance) * 10.0

        # Append current action to previous actions
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
