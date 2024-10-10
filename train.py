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
import torch
from torch import nn
import argparse
import os


set_random_seed(42)
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="ee_position_int_reward_obs_with_depth_v4", help='Name of the run')
parser.add_argument('--load_model', type=str, help='Path to the model to load', default="")
args = parser.parse_args()

run_name = args.run_name
load_model = args.load_model


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# Define a custom feature extractor for the image
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Image has shape (256, 256, 3), assuming that's the key 'image'
        n_input_channels = observation_space.spaces['image'].shape[2]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        
        # Compute shape by doing a forward pass with dummy data
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.spaces['image'].sample()[None]).float()).shape[1]

        # Combine the CNN output and end-effector position (3D)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 3, features_dim),  # 3 = number of end-effector position dimensions
            nn.ReLU()
        )

    def forward(self, observations):
        # Process the image part of the observation through the CNN
        image = observations['image'].float() / 255.0
        image_features = self.cnn(image)
        
        # Get the end-effector position
        end_effector_pos = observations['end_effector_pos'].float()
        
        # Concatenate the CNN output and the end-effector position
        combined_features = torch.cat((image_features, end_effector_pos), dim=1)
        
        return self.linear(combined_features)


name = run_name
run = wandb.init(
    project="EE_states_added",
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

log_dir = f"./new_obs_results/{name}"
os.makedirs(log_dir, exist_ok=True)

my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

total_timesteps = 1000000
callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

policy_kwargs = dict(activation_fn=torch.nn.ReLU, 
                     net_arch=dict(pi=[64, 64], qf=[400, 300]),
                     features_extractor_class=CustomCNN,
                     features_extractor_kwargs=dict(features_dim=256))  # Output of feature extractor)

# Load the model if a path is provided, otherwise create a new model
if load_model and os.path.exists(load_model):
    model = SAC.load(load_model, env=my_env, verbose=1)
    print(f"Loaded model from {load_model}")
else:
    model = SAC(
        "MultiInputPolicy",
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
