import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp
import time
import gc
import ikpy.chain
import os
import omni

def euler_angles_to_quat(euler_angles, order="xyz"):
    """
    Convert Euler angles to a quaternion.

    Args:
        euler_angles (list or np.ndarray): Euler angles [roll, pitch, yaw] in radians.
        order (str): Order of rotation axes. Default is 'xyz'.

    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    roll, pitch, yaw = euler_angles

    # Calculate trigonometric values
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Compute quaternion components
    if order == "xyz":
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    elif order == "zyx":
        w = cr * cp * cy + sr * sp * sy
        x = cr * sp * sy - sr * cp * cy
        y = cr * sp * cy + sr * cp * sy
        z = sr * sp * cy - cr * cp * sy
    else:
        raise ValueError(f"Unsupported order: {order}")

    return np.array([w, x, y, z])


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

        self.episode_length = 0
        self.initial_angles = np.deg2rad([-90, -110, 120, -120, 180, 0]) # -90, 30, 120, -120, 0, 0
        # Initialize list to store previous actions for intrinsic reward calculation
        self.tic = time.time()
        self.max_angles = np.deg2rad([-80, 30, 140, -90, 50+180])
        self.min_angles = np.deg2rad([-120, -30, 100, -150, -50+180])
        self.last_angles = self.initial_angles[:5]
    
    def get_reward(self, rgb_obs, depth_obs, ee_location):
        """
        Calculate rewards based on lexicographic ordering:
        1. Primary Objective: Maximize cube visibility.
        2. Secondary Objective: Minimize plant visibility.
        """
        rgb_obs_resized = cv2.resize(rgb_obs, (depth_obs.shape[1], depth_obs.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Define cube and plant masks
        cube_mask = (depth_obs >= 0.28) & (depth_obs <= 0.44)
        plant_mask = (depth_obs >= 0.01) & (depth_obs <= 0.27)

        # Calculate pixel counts for cube and plant
        cube_pixels = np.sum(cube_mask)
        plant_pixels = np.sum(plant_mask)

        # Lexicographic rewards
        if cube_pixels >= 2000:  # Example threshold for sufficient cube visibility
            # Secondary objective: Minimize plant visibility
            reward = -plant_pixels / 63000.0
        else:
            # Primary objective: Focus on increasing cube visibility
            reward = cube_pixels / 2500.0

        # Save images at intervals for debugging
        if time.time() - self.tic > 2.0:
            cv2.imwrite("/maniRL/images/cube.png", rgb_obs_resized * cube_mask[..., None])
            cv2.imwrite("/maniRL/images/plant.png", rgb_obs_resized * plant_mask[..., None])
            self.tic = time.time()

        return reward

    def step(self, action):
        action = self.last_angles + action * 0.1
        action = np.clip(action, self.min_angles, self.max_angles)
        self.robot.send_angles(0, np.append(action, 0.0), degrees=False)
        self.last_angles = action

        self.world._world.step()
        rgb_obs, depth_obs = self.get_observation()
        ee_location = np.array(self.robot.get_ee_position())

        # Calculate reward
        reward = self.get_reward(rgb_obs, depth_obs, ee_location)

        # Update state
        self.episode_length += 1
        truncated = self.is_truncated()
        observation = {
            'image': rgb_obs,
            'end_effector_pos': ee_location
        }

        # Return lexicographic-based reward
        return observation, float(reward), False, truncated, {}


    
    def get_observation(self):
        rgb, depth = self.world.get_image()
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth


    def is_truncated(self):
        if self.episode_length >= self._max_episode_length:
            return True
        return False
    
    def reset(self, seed=None, *args, **kwargs):
        self.world._world.reset()
        # initial_angles = np.deg2rad([-90, np.random.uniform(-110, 40), 120, -120, 0, 0]) # -90, 30, 120, -120, 0, 0
        self.world.goal_cube.set_world_pose([np.random.uniform(-0.2, 0.2), 
                                             -0.4, 
                                             np.random.uniform(0.22, 0.4)], [0, 0, 0, 1])
        self.world.goal_cube_2.set_world_pose([np.random.uniform(-0.3, 0.3), 
                                             -0.4, 
                                             np.random.uniform(0.15, 0.4)], [0, 0, 0, 1])
        self.world.goal_cube_3.set_world_pose([np.random.uniform(-0.4, 0.4), 
                                             -0.4, 
                                             np.random.uniform(0.15, 0.4)], [0, 0, 0, 1])
        
        first_joint = np.random.uniform(-110, -80)
        if np.random.uniform() < 0.2:
            second_joint = np.random.uniform(10, 30)
        else:
            second_joint = np.random.uniform(-30, 0)
        self.robot.send_angles(0, np.array([first_joint, second_joint, 120, 0, 0, 0]), degrees=True)
        for i in range(30):
            self.world._world.step()
        init_angles = np.deg2rad([first_joint, second_joint, 120, -120, 180, 0])
        self.robot.send_angles(0, init_angles, degrees=False)
        for i in range(30):
            self.world._world.step()

        self.episode_length = 0
        self.last_angles = init_angles[:5]
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
