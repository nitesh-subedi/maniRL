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
from stable_baselines3.ppo import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import torch as th
import argparse


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

log_dir = "./cnn_policy"
# set headles to false to visualize training
my_env = maniEnv(config=CONFIG)
check_env(my_env)

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(vf=[128, 128, 128], pi=[128, 128, 128]))
policy = CnnPolicy
total_timesteps = 500000

if args.test is True:
    total_timesteps = 10000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="franka_policy_cont_checkpoint")
model = PPO.load(
    "./cnn_policy/jetbot_policy.zip", 
    env=my_env, 
    device="cuda:0",
    tensorboard_log=log_dir
    )

model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/franka_policy_cont")

my_env.close()
