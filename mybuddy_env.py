import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
import time
import gc


class MyBuddyEnv(gym.Env):
    def __init__(self,
                 skip_frame=1,
                 physics_dt=1.0 / 60.0,
                 rendering_dt=1.0 / 60.0,
                 max_episode_length=200,
                 seed=0,
                 config={"headless": True, "anti_aliasing": 0}) -> None:
        super().__init__()
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

        # # Combine both spaces using Tuple or Dict (depending on your needs)
        self.observation_space = spaces.Dict({
            'image': image_obs_space,
            'end_effector_pos': end_effector_space
        })
        # self.observation_space = image_obs_space

        self.lower_red = np.array([90, 50, 50])
        self.upper_red = np.array([130, 255, 255])

        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255]) 

        self.episode_length = 0
        self.initial_angles = np.deg2rad([-90, -110, 120, -120, 180, 0]) # -90, 30, 120, -120, 0, 0
        # Initialize list to store previous actions for intrinsic reward calculation
        self.tic = time.time()
        self.max_angles = np.deg2rad([-80, 40, 130, -90, 50+180])
        self.min_angles = np.deg2rad([-100, 0, 100, -130, -50+180])
        self.last_angles = self.initial_angles[:5]


    def step(self, action):
        action = self.last_angles + action * 0.1
        action = np.clip(action, self.min_angles, self.max_angles)
        # print(np.rad2deg(action))
        self.robot.send_angles(0, np.append(action, 0.0), degrees=False)
        self.world._world.step()
        self.last_angles = action
        rgb_obs, depth_obs = self.get_observation()
        rgb_obs = cv2.cvtColor(rgb_obs, cv2.COLOR_BGR2RGB)
        # rgb_obs = np.zeros_like(rgb_obs)
        ee_location = np.array(self.robot.get_ee_position())
        # print(ee_location)
        reward = self.get_reward(rgb_obs, depth_obs)
        # Combine rewards
        total_reward = reward #+ int_reward
        self.episode_length += 1
        if self.episode_length % 100 == 0:
            gc.collect()
        truncated = self.is_truncated()
        observation = {
            'image': rgb_obs,
            'end_effector_pos': ee_location
        }
        return observation, float(total_reward), False, truncated, {}
    
    def get_observation(self):
        rgb, depth = self.world.get_image()
        return rgb, depth

    def get_reward(self, rgb_obs, depth_obs):
        rgb_obs_resized = cv2.resize(rgb_obs, (depth_obs.shape[1], depth_obs.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Define cube and plant masks
        cube_mask = (depth_obs >= 0.28) & (depth_obs <= 0.44)
        plant_mask = (depth_obs >= 0.01) & (depth_obs <= 0.27)

        # Apply depth masks to the RGB image
        rgb_cube = rgb_obs_resized.copy()
        rgb_cube[~cube_mask] = 0

        rgb_plant = rgb_obs_resized.copy()
        rgb_plant[~plant_mask] = 0

        # Ensure masks are boolean and apply them properly
        cube_mask = cube_mask.astype(np.uint8)
        plant_mask = plant_mask.astype(np.uint8)
        rgb_cube_binary = (rgb_cube > 0).astype(np.uint8)
        rgb_plant_binary = (rgb_plant > 0).astype(np.uint8)

        # Calculate wanted and unwanted pixels
        wanted_pixels = np.sum(cube_mask & cv2.cvtColor(rgb_cube_binary, cv2.COLOR_RGB2GRAY))
        unwanted_pixels = np.sum(plant_mask & cv2.cvtColor(rgb_plant_binary, cv2.COLOR_RGB2GRAY))

        # Calculate reward
        # reward = -(unwanted_pixels / 63000)
        reward = (wanted_pixels / 2500) * 0.2

        # Save images at intervals
        if time.time() - self.tic > 2.0:
            # print(-(unwanted_pixels / 63000), (wanted_pixels / 2500) * 0.2)
            cv2.imwrite("/maniRL/images/cube.png", rgb_cube)
            cv2.imwrite("/maniRL/images/plant.png", rgb_plant)
            self.tic = time.time()
        # Clean up
        del unwanted_pixels, plant_mask, rgb_plant
        return reward

    def is_truncated(self):
        if self.world._world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            return True
        return False
    
    def reset(self, seed=None):
        self.world._world.reset()
        # initial_angles = np.deg2rad([-90, np.random.uniform(-110, 40), 120, -120, 0, 0]) # -90, 30, 120, -120, 0, 0
        init_angles = np.deg2rad([-90, 0, 120, -120, 180, 0])
        self.robot.send_angles(0, init_angles, degrees=False)
        self.world.goal_cube.set_world_pose([np.random.uniform(-0.2, 0.2), 
                                             -0.4, 
                                             np.random.uniform(0.22, 0.4)], [0, 0, 0, 1])
        # self.world.goal_cube_2.set_world_pose([np.random.uniform(-0.3, 0.3), 
        #                                      -0.4, 
        #                                      np.random.uniform(0.22, 0.4)], [0, 0, 0, 1])
        # self.world.goal_cube_3.set_world_pose([np.random.uniform(-0.3, 0.3), 
        #                                      -0.4, 
        #                                      np.random.uniform(0.22, 0.4)], [0, 0, 0, 1])
        for i in range(30):
            self.world._world.step()
        self.episode_length = 0
        self.last_angles = init_angles[:5]
        # self.previous_actions = []
        self.w = {}
        self.observing_cube = False
        self.total_visits = 0
        ee_location = np.array(self.robot.get_ee_position())
        rgb_obs, depth_obs = self.get_observation()
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


