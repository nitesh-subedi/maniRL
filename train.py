from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.ppo import CnnPolicy as ppocnn
from stable_baselines3.sac import CnnPolicy as saccnn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import torch as th
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed

seed_number = 40
set_random_seed(seed_number)

name = f"SAC_seed_{seed_number}_ent_auto"
run = wandb.init(
    project="plant_manipulation",
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

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = f"./results/{name}"

# set headless to false to visualize training
my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(vf=[128, 128, 128], pi=[128, 128, 128]))
total_timesteps = 100000

# if args.test is True:
#     total_timesteps = 10000

policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=th.nn.ReLU,
)

# callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")
# callback = SaveOnBestTrainingRewardCallback(save_freq=10000, save_path=log_dir, name_prefix="franka_policy")
# model = PPO(
#     ppocnn,
#     my_env,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     n_steps=5120,
#     batch_size=64,
#     learning_rate=0.000125,
#     gamma=0.9,
#     ent_coef=7.5e-08,
#     clip_range=0.3,
#     n_epochs=10,
#     gae_lambda=1.0,
#     max_grad_norm=0.9,
#     vf_coef=0.95,
#     device="cuda:0",
#     tensorboard_log=f"{log_dir}/runs/{run.id}",
# )


model = SAC(
    saccnn,  # You can replace "MlpPolicy" with your custom policy if needed
    my_env,
    verbose=1,
    buffer_size=100000,  # Replay buffer size
    batch_size=256,  # Same batch size as PPO
    learning_rate=0.000125,  # Same learning rate as PPO
    gamma=0.99,  # Same gamma as PPO
    ent_coef='auto',  # Similar entropy coefficient (note: SAC has automatic entropy tuning by default)
    tau=0.005,  # Target smoothing coefficient (default: 0.005)
    train_freq=(1, "episode"),  # Training frequency (default: (1, "episode"))
    gradient_steps=-1,  # Number of gradient steps (default: -1, i.e., train for every step taken in the environment)
    learning_starts=100,  # Number of steps before learning starts (default: 100)
    use_sde=False,  # Whether to use State Dependent Exploration (default: False)
    device="cuda:0",  # Use GPU for training
    tensorboard_log=f"{log_dir}/tensorboard",  # Uncomment and set log directory if using tensorboard
)

model.learn(
    total_timesteps=total_timesteps,
    callback=[WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/models/{run.id}",
        verbose=2), callback]
)

run.finish()

my_env.close()
