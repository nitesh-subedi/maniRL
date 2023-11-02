# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from env import maniEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import torch as th
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="sb3",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
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

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = "./results"


# set headles to false to visualize training
my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(vf=[128, 128, 128], pi=[128, 128, 128]))
policy = CnnPolicy
total_timesteps = 100000

if args.test is True:
    total_timesteps = 10000

# callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
# callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="franka_policy_checkpoint")
# callback = SaveOnBestTrainingRewardCallback(save_freq=10000, save_path=log_dir, name_prefix="franka_policy")
model = PPO(
    policy,
    my_env,
    # policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=5120,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=10,
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    device="cuda:0",
    tensorboard_log=f"{log_dir}/runs/{run.id}",
)
model.learn(
    total_timesteps=total_timesteps, 
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/models/{run.id}",
        verbose=2,
        )
    )

run.finish()

my_env.close()
