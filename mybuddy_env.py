import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
import time
import gc
import os
from mybuddy.utils import DepthEstimator


class MyBuddyEnv(gym.Env):
    def __init__(self,
                 skip_frame=1,
                 physics_dt=1.0 / 60.0,
                 rendering_dt=1.0 / 60.0,
                 max_episode_length=200,
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
        self.world.initialise(usd_path="/isaac_sim_assets/plant_v21/plant_v21.usd", 
                              env_usd = "/isaac_sim_assets/env_v2/environment.usd",
                              hdr_path="/isaac_sim_assets/env_v2/textures/rosendal_plains_2_4k.hdr")

        from mybuddy.robot import Robot
        self.robot = Robot(
            urdf_path="/isaac_sim_assets/urdf/urdf.urdf",
            world=self.world.world, simulation_app=self._simulation_app)

        self.world._world.step()
        self._simulation_app.update()

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
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
        self.episode_length = 0
        self.initial_angles = np.deg2rad([-90, -110, 120, -120, 0, 0]) # -90, 30, 120, -120, 0, 0
        # Initialize list to store previous actions for intrinsic reward calculation
        self.tic = time.time()
        # self.ikchain = ikpy.chain.Chain.from_urdf_file("/isaac_sim_assets/urdf/urdf.urdf")
        os.makedirs("/maniRL/images", exist_ok=True)
        self.last_actions = np.zeros(2)
        self.min_angles = np.deg2rad([-120, -10])
        self.max_angles = np.deg2rad([-80, 40])


    def step(self, action):
        action = np.clip(action, -1.0, 1.0) * 0.1
        # action = self.denormalize_action(action)
        action = self.last_actions + action
        print(np.rad2deg(action))
        action = np.clip(action, self.min_angles, self.max_angles)

        self.robot.send_angles(0, np.append(action, np.deg2rad([120, -120, 0, 0])), degrees=False)
        self.last_actions = action
        self.world._world.step()
        rgb_obs, depth_obs = self.get_observation()
        ee_location = np.array(self.robot.get_ee_position())
        reward = self.get_reward(depth_obs, rgb_obs)
        # int_reward = self.get_intrinsic_reward(action)
        # Combine rewards
        total_reward = reward  # + int_reward
        done = self.get_done()
        self.episode_length += 1
        if self.episode_length % 100 == 0:
            gc.collect()
        truncated = self.is_truncated()
        observation = {
            'image': rgb_obs,
            'end_effector_pos': ee_location
        }
        self.last_reward = total_reward
        return observation, float(total_reward), done, truncated, {}

    @staticmethod
    def get_done():
        return False
    
    def get_observation(self):
        rgb_obs = self.world.get_image()
        return rgb_obs


    def get_reward(self, rgb_obs):
        # Load and preprocess the image
        img = cv2.cvtColor(rgb_obs, cv2.COLOR_BGR2RGB)
        input_batch = self.depth_estimator.remove_nearest_objects(img)
        # # Resize depth_obs to match rgb_obs dimensions
        # rgb_obs_resized = cv2.resize(rgb_obs, (depth_obs.shape[1], depth_obs.shape[0]), interpolation=cv2.INTER_NEAREST)

        # # Define cube and plant masks
        # # cube_mask = (depth_obs_resized >= 0.28) & (depth_obs_resized <= 0.44)
        # # depth_for_cube = depth_obs_resized.copy()
        # # depth_for_cube[~cube_mask] = 0
        # # depth_for_cube[cube_mask] = 1
        # # wanted_pixels = np.sum(depth_for_cube)

        # plant_mask = (depth_obs >= 0.01) & (depth_obs <= 0.27)
        # depth_for_plant = depth_obs.copy()
        # depth_for_plant[~plant_mask] = 0
        # depth_for_plant[plant_mask] = 1
        # unwanted_pixels = np.sum(depth_for_plant)

        # # Calculate reward
        # reward = -(unwanted_pixels / 63000) * 3
        # # reward += (wanted_pixels / 2500)

        # # Apply depth masks to the RGB image
        # # rgb_cube = rgb_obs.copy()
        # # rgb_cube[~cube_mask] = 0

        # rgb_plant = rgb_obs_resized.copy()
        # rgb_plant[~plant_mask] = 0

        # # Save images at intervals
        # if time.time() - self.tic > 2.0:
        #     # print(-(unwanted_pixels / 63000) * 2, wanted_pixels / 2500)
        #     # cv2.imwrite("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/cube.png", cv2.cvtColor(rgb_cube, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite("/maniRL/images/plant.png", cv2.cvtColor(rgb_plant, cv2.COLOR_RGB2BGR))
        #     self.tic = time.time()

        # Clean up
        del unwanted_pixels, plant_mask, depth_for_plant, rgb_plant
        return reward


    
    def is_truncated(self):
        if self.world._world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            return True
        return False
    
    def reset(self, seed=None):
        self.world._world.reset()
        # initial_angles = np.deg2rad([-90, np.random.uniform(-110, 40), 120, -120, 0, 0]) # -90, 30, 120, -120, 0, 0
        init_angles = np.deg2rad([-90, np.random.uniform(-20, 30), 120, -120, 0, 0])
        self.robot.send_angles(0, init_angles, degrees=False)
        for i in range(30):
            self.world._world.step()
        self.world.goal_cube.set_world_pose([np.random.uniform(-0.02, 0.02), -0.4, 0.22], [0, 0, 0, 1])
        self.episode_length = 0
        self.w = {}
        self.observing_cube = False
        self.total_visits = 0
        ee_location = np.array(self.robot.get_ee_position())
        rgb_obs, depth_obs = self.get_observation()
        self.last_actions = np.zeros(2)
        observation = {
            'image': rgb_obs,
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


