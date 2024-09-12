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


# # Custom Callback for Saving Models
# class CustomCallback(BaseCallback):
#     def __init__(self, save_freq: int, save_path: str, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = save_path

#     def _init_callback(self) -> None:
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.save_freq == 0:
#             model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
#             self.model.save(model_path)
#             if self.verbose > 0:
#                 print(f"Saving model checkpoint to {model_path}")

#         # Log additional info
#         if self.locals and 'infos' in self.locals:
#             infos = self.locals['infos']
#             if infos and len(infos) > 0:
#                 current_step = infos[0].get('current_step', 0)
#                 distance_to_goal = infos[0].get('distance_to_goal', 0)
#                 self.logger.record('info/current_step', current_step)
#                 self.logger.record('info/distance_to_goal', distance_to_goal)
        
#         return True

# seed_number = 42
set_random_seed(0)

name = f"previous_model_finetuning"
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

log_dir = f"./results/{name}"

# set headless to false to visualize training
my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(vf=[128, 128, 128], pi=[128, 128, 128]))
total_timesteps = 1000000

callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64, 64], qf=[800, 600]))

# model = SAC(
#     saccnn,  # You can replace "MlpPolicy" with your custom policy if needed
#     my_env,
#     verbose=1,
#     policy_kwargs=policy_kwargs,
#     buffer_size=150000,  # Replay buffer size
#     gamma=0.9,
#     device="cuda:0",
#     tensorboard_log=f"{log_dir}/tensorboard",
# )
# model = PPO(
#     ppocnn,
#     my_env,
#     batch_size=256,
#     gamma=0.9,
#     ent_coef=0.008,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     tensorboard_log=f"{log_dir}/tensorboard",)

model = SAC.load("/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/results/SAC_new_bonus_lr_default/mybuddy_policy_checkpoint_1000000_steps.zip", env=my_env, verbose=1)

model.learn(
    total_timesteps=total_timesteps,
    callback=[WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/models/{run.id}",
        verbose=2), callback]
)

run.finish()

my_env.close()
