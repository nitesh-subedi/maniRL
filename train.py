from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.ppo import CnnPolicy as ppocnn
from stable_baselines3.sac import CnnPolicy as saccnn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import torch as th
import time
import argparse
import os


set_random_seed(42)
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="SAC_with_script", help='Name of the run')
parser.add_argument('--load_model', type=str, help='Path to the model to load', default=None)
args = parser.parse_args()

run_name = args.run_name
load_model = args.load_model


name = run_name
run = wandb.init(
    project="Real_Plant_SAC",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
    id=name,  # optional,
)


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

log_dir = f"./results/SAC_with_script/{name}"
os.makedirs(log_dir, exist_ok=True)

my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

total_timesteps = 10010
callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64, 64], qf=[400, 300]))

# Load the model if a path is provided, otherwise create a new model
if load_model and os.path.exists(load_model):
    model = SAC.load(load_model, env=my_env, verbose=1)
    print(f"Loaded model from {load_model}")
else:
    model = SAC(
        saccnn,
        my_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        buffer_size=100000,
        gamma=0.9,
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

my_env.close()
