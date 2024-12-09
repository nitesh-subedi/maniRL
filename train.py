from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import torch
import argparse
import os
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


set_random_seed(42)
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="SAC_high_level_v90", help='Name of the run')
parser.add_argument('--load_model', type=str, help='Path to the model to load', default="")
args = parser.parse_args()

run_name = args.run_name
load_model = args.load_model

name = run_name
run = wandb.init(
    project="New_formulation",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
    id=name,  # optional,
)

CONFIG = {
    "width": 1280,  # Reduce resolution width
    "height": 720,  # Reduce resolution height
    "window_width": 1280,  # Window width (can match rendering resolution for consistency)
    "window_height": 720,  # Window height (can match rendering resolution for consistency)
    "headless": True,  # Keep headless mode enabled for non-GUI rendering
    "renderer": "RayTracedLighting",  # Switch to a faster rendering mode than RayTracedLighting
    "display_options": 0,  # Disable display options to remove extra elements (e.g., grid)
    "anti_aliasing": 0,  # Keep anti-aliasing disabled to improve performance
    "enable_gpu_optimizations": True,  # Custom field to suggest GPU optimizations
    "reduce_material_complexity": True,  # Placeholder to suggest reducing material/lighting
}


log_dir = f"/maniRL/new_obs_results/{name}"
os.makedirs(log_dir, exist_ok=True)

my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

total_timesteps = 1000000
callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

policy_kwargs = dict(activation_fn=torch.nn.Tanh, 
                     net_arch=dict(pi=[64, 64], qf=[400, 300]))

action_n = my_env.action_space.shape[0]

action_noise = NormalActionNoise(mean=np.zeros(action_n), sigma=0.2 * np.ones(action_n))

# Load the model if a path is provided, otherwise create a new model
if load_model and os.path.exists(load_model):
    model = SAC.load(load_model, env=my_env, verbose=1,
                     buffer_size=100000,
                    batch_size=256,
                    gamma=0.9,
                    action_noise=action_noise,
                    device="cuda:0",
                    tensorboard_log=f"{log_dir}/tensorboard")
    print(f"Loaded model from {load_model}")

else:
    model = SAC(
        "MultiInputPolicy",
        my_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        buffer_size=100000,
        batch_size=256,
        gamma=0.8,
        action_noise=action_noise,
        device="cuda:0",
        tensorboard_log=f"{log_dir}/tensorboard",
    )

model.learn(
    total_timesteps=total_timesteps,
    callback=[callback, WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/models/{run.id}",
        verbose=2)],
)

# save the replay buffer
model.save_replay_buffer(f"{log_dir}")

my_env.close()
