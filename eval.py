from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC, PPO
import time
from stable_baselines3.sac.policies import SACPolicy

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
policy_path = "/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/results/SAC_with_script/ee_position_rewad_v2/mybuddy_policy_checkpoint_40000_steps.zip"

my_env = maniEnv(config=CONFIG)
model = SAC.load(policy_path)
obs, info = my_env.reset()

for _ in range(20):
    obs, info = my_env.reset()
    terminated = False
    truncated = False
    done = terminated or truncated
    time.sleep(1.0)
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        print('\n')
        obs, reward, terminated, truncated, info = my_env.step(actions)
        done = terminated or truncated  
        print(f"Reward: {reward}")
        time.sleep(0.1)

my_env.reset()
my_env.close()
time.sleep(5.0)
        
