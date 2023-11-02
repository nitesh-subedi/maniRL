from env import maniEnv
from stable_baselines3 import PPO

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Set display options to show default grid
    "anti_aliasing": 0,
}


# Choose the policy path to visualize
policy_path = "./cnn_policy/franka_policy_cont_checkpoint_290000_steps.zip"

my_env = maniEnv(config=CONFIG)
model = PPO.load(policy_path)
obs, info = my_env.reset()

for _ in range(5):
    terminated = False
    truncated = False
    done = terminated or truncated
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        print(actions)
        print('\n')
        obs, reward, terminated, truncated, info = my_env.step(actions)
