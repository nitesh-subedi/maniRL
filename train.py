from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import torch
import torch as th
from torch import nn
import argparse
import os
import torchvision.models as models
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


action_noise = NormalActionNoise(mean=np.zeros(5), sigma=0.03 * np.ones(5))

set_random_seed(42)
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="SAC_multi_v30", help='Name of the run')
parser.add_argument('--load_model', type=str, help='Path to the model to load', default="")
args = parser.parse_args()

run_name = args.run_name
load_model = args.load_model


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define a custom feature extractor for the image
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Use ResNet (pre-trained or untrained) as the feature extractor for image
                resnet = models.resnet18(pretrained=True)
                # Remove the final classification layer, keeping the feature extractor part
                resnet = nn.Sequential(*list(resnet.children())[:-1])
                for param in resnet.parameters():
                    param.requires_grad = False

                extractors[key] = nn.Sequential(
                    resnet,
                    nn.Flatten()
                )
                # The output size of resnet18 is 512
                total_concat_size += 512
            elif key == "end_effector_pos":
                # For the vector input, we'll use a simple MLP with 16 output units
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16),
                    nn.ReLU()
                )
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)


name = run_name
run = wandb.init(
    project="EE_states_added",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
    id=name,  # optional,
)


# CONFIG = {
#     "width": 1280,
#     "height": 720,
#     "window_width": 1920,
#     "window_height": 1080,
#     "headless": True,
#     "renderer": "RayTracedLighting",
#     "display_options": 3286,  # Set display options to show default grid
#     "anti_aliasing": 0,
# }

CONFIG = {
    "width": 1280/3,  # Reduce resolution width
    "height": 720/3,  # Reduce resolution height
    "window_width": 1280/3,  # Window width (can match rendering resolution for consistency)
    "window_height": 720/3,  # Window height (can match rendering resolution for consistency)
    "headless": True,  # Keep headless mode enabled for non-GUI rendering
    "renderer": "RayTracedLighting",  # Switch to a faster rendering mode than RayTracedLighting
    "display_options": 0,  # Disable display options to remove extra elements (e.g., grid)
    "anti_aliasing": 0,  # Keep anti-aliasing disabled to improve performance
    "enable_gpu_optimizations": True,  # Custom field to suggest GPU optimizations
    "reduce_material_complexity": True,  # Placeholder to suggest reducing material/lighting
}


log_dir = f"./new_obs_results/{name}"
os.makedirs(log_dir, exist_ok=True)

my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

total_timesteps = 1000000
callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

policy_kwargs = dict(activation_fn=torch.nn.ReLU, 
                     net_arch=dict(pi=[64, 64], qf=[400, 300]),
                     features_extractor_class=CustomCombinedExtractor)

# Load the model if a path is provided, otherwise create a new model
if load_model and os.path.exists(load_model):
    model = SAC.load(load_model, env=my_env, verbose=1,
                     buffer_size=100000,
                    batch_size=512,
                    gamma=0.8,
                    device="cuda:0",
                    action_noise=action_noise,
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
        gamma=0.9,
        device="cuda:0",
        action_noise=action_noise,
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
