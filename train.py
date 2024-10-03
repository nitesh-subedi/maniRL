from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.sac import CnnPolicy as saccnn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import torch as th
import argparse
import os


set_random_seed(42)
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="her_reward_v4", help='Name of the run')
parser.add_argument('--load_model', type=str, help='Path to the model to load', default="/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/real_env_results/her_reward_v3_run_1/mybuddy_policy_checkpoint_10000_steps.zip")
parser.add_argument('--load_replay_buffer', type=str, help='Path to the replaybuffer to load', default="/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/real_env_results/her_reward_v3_run_1/replay_buffer.pkl")

args = parser.parse_args()

run_name = args.run_name
load_model = args.load_model
replay_buffer = args.load_replay_buffer


name = run_name
run = wandb.init(
    project="Plant Manipulation Real Environment",
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

log_dir = f"./real_env_results/{name}"
os.makedirs(log_dir, exist_ok=True)

my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

total_timesteps = 1000000
callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64, 64], qf=[400, 300]))
goal_selection_strategy = "future"
# Load the model if a path is provided, otherwise create a new model
if load_model and os.path.exists(load_model):
    model = SAC.load(load_model, env=my_env, verbose=1)
    model.load_replay_buffer(replay_buffer)
    print(f"Loaded model from {load_model}")
else:
    model = SAC(
        "MultiInputPolicy",
        my_env,
        verbose=1,
        learning_starts=1000,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy),
        buffer_size=100000,
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
model.save_replay_buffer(f"{log_dir}/replay_buffer")

my_env.close()
