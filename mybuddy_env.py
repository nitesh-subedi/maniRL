import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
import time
from collections import deque
import gc
from mybuddy.utils import DepthEstimator
from memory_profiler import profile


class MyBuddyEnv(gym.Env):
    def __init__(self,
                 skip_frame=1,
                 physics_dt=1.0 / 60.0,
                 rendering_dt=1.0 / 60.0,
                 max_episode_length=1000,
                 seed=0,
                 config={"headless": True, "anti_aliasing": 0}) -> None:
        super().__init__()
        self.depth_estimator = DepthEstimator()
        self._simulation_app = SimulationApp(launch_config=config)
        self._simulation_app.set_setting("/app/window/drawMouse", True)
        self._simulation_app.set_setting("/app/livestream/proto", "ws")
        self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        self._simulation_app.set_setting("/ngx/enabled", False)

        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.services.streamclient.webrtc")
        enable_extension("omni.syntheticdata")

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

        self.world._world.step()
        self._simulation_app.update()

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(5,), dtype=np.float32)
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

        # # Combine both spaces using Tuple or Dict (depending on your needs)
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
        self.initial_angles = np.deg2rad([-90, -110, 120, -120, 0, 0]) # -90, 30, 120, -120, 0, 0
        # Initialize list to store previous actions for intrinsic reward calculation
        max_length = 10000
        self.previous_actions = deque(maxlen=max_length)
        self.tic = time.time()
        self.last_reward = 0
        self.first_call = True


    def step(self, action):
        action = np.clip(action, -5.0, 5.0) / 5.0
        action = np.array([action[0], action[1], action[2], action[3], action[4], 0.0]) * 0.1 
        action = self.last_angles + action
        self.robot.send_angles(0, action, degrees=False)
        self.world._world.step()
        collision, end_effector_collision = self.robot.check_collision()
        self.last_angles = action
        obs = self.get_observation()
        ee_location = np.array(self.robot.get_ee_position())
        reward = self.get_reward(obs, collision, ee_location)
        int_reward = self.get_intrinsic_reward(action)
        # Combine rewards
        total_reward = reward + int_reward
        if end_effector_collision:
            total_reward -= 2

        done = self.get_done(collision)
        self.episode_length += 1
        if self.episode_length % 100 == 0:
            gc.collect()
        truncated = self.is_truncated()
        observation = {
            'image': obs,
            'end_effector_pos': ee_location
        }
        self.last_reward = total_reward
        return observation, float(total_reward), done, truncated, {}

    @staticmethod
    def get_done(collision):
        if collision:
            return True
        return False
    
    def get_observation(self):
        return cv2.resize(self.world.get_image(), (256, 256))

    def get_reward(self, obs, collision, ee_location):
        # Compute reward based on the observation
        unwanted_pixels, result = self.depth_estimator.remove_nearest_objects(obs, threshold=0.6)
        if time.time() - self.tic > 5:
            self.tic = time.time()
            cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_goal_output.jpg", result)
            cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/real_output.jpg", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        # More reward if less unwanted pixels
        reward = - (unwanted_pixels / 60000) * 30
        del unwanted_pixels
        del result

        reward += - (ee_location[1] / 2) * 100

        # Penalize collision
        if collision:
            reward -= 20

        return reward + 0.1

    
    def is_truncated(self):
        if self.world._world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            return True
        return False
    
    def reset(self, seed=None):
        if self.first_call:
            self.world._world.reset()
            self.first_call = False
        else:
            self.world._world.reset(soft=True)
        # initial_angles = np.deg2rad([-90, np.random.uniform(-110, 40), 120, -120, 0, 0]) # -90, 30, 120, -120, 0, 0
        self.robot.send_angles(0, np.deg2rad([-90, np.random.uniform(-110, 30), 120, -120, 0, 0]), degrees=False)
        for i in range(30):
            self.world._world.step()
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

