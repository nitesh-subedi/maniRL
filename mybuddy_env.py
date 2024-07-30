import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from isaacsim import SimulationApp


class MyBuddyEnv(gym.Env):
    def __init__(self,
                 skip_frame=1,
                 physics_dt=1.0 / 60.0,
                 rendering_dt=1.0 / 60.0,
                 max_episode_length=512,
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

        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        # Import world
        from world.world import SimulationEnv
        self.world = SimulationEnv(config={"physics_dt": physics_dt, "rendering_dt": rendering_dt})
        self.world.initilise()

        from mybuddy.robot import Robot
        self.robot = Robot(
            urdf_path="/home/nitesh/workspace/rosws/mybuddy_robot_rl/src/mybuddy_description/urdf/urdf.urdf",
            world=self.world._world, simulation_app=self._simulation_app)

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

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        action = np.array([action[0], action[1], action[2], action[3], action[4], 0.0]) * 0.1
        action = self.last_angles + action
        self.robot.send_angles(0, action, degrees=False)
        collision = self.robot.check_collision()
        self.last_angles = action
        for i in range(5):
            self.world._world.step()
        obs = self.get_observation()
        reward = self.get_reward(obs, collision)
        done = self.get_done(collision)
        truncated = self.is_truncated()

        return obs, float(reward), done, truncated, {}

    @staticmethod
    def get_done(collision):
        if collision:
            return True
        return False

    def get_observation(self):
        self.world._world.render()
        image = self.world.get_image()
        image = cv2.resize(image, (256, 256))
        return image

    def get_reward(self, obs, collision):
        camera_frame = obs
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

        # Create a mask for green color
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(image, image, mask=mask)

        reward = cv2.countNonZero(mask)

        if collision:
            reward -= 1000
        return reward + 10

    def is_truncated(self):
        if self.world._world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            return True
        return False

    def reset(self, seed=None):
        self.world._world.reset()
        self.world._world.reset()
        # Set the goal cube to a random position around [0, -0.4, 0.22]
        self.world.goal_cube.set_world_pose([0.0, -0.4, 0.22], [0, 0, 0, 1])
        return self.get_observation(), {}

    def render(self, mode="human"):
        pass

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
