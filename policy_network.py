from stable_baselines3.sac.policies import SACPolicy
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ContinuousCritic


class MultiActorPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MultiActorPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        # CNN for 'image' observations
        self.image_actor = self._build_cnn_actor(self.observation_space['image'].shape)

        # Fully connected network for 'end_effector_pos'
        self.ee_actor = self._build_mlp_actor(self.observation_space['end_effector_pos'].shape)

        # Common output layer to combine actions
        self.action_combiner = nn.Sequential(
            nn.Linear(2 * self.action_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.shape[0])
        )

    def _build_cnn_actor(self, input_shape):
        """
        CNN-based actor for image observations.
        """
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self._cnn_output_size(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.shape[0])
        )

    def _cnn_output_size(self, input_shape):
        dummy_input = th.zeros(1, *input_shape)  # Batch size = 1
        output = self._build_cnn_layers(input_shape)(dummy_input)
        return output.view(-1).size(0)

    def _build_cnn_layers(self, input_shape):
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )

    def _build_mlp_actor(self, input_shape):
        """
        MLP-based actor for end-effector position observations.
        """
        return nn.Sequential(
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.shape[0])
        )

    # def forward(self, obs, deterministic=False):
    #     image_obs = obs['image']
    #     ee_obs = obs['end_effector_pos']

    #     # Actor outputs
    #     image_action = self.image_actor(image_obs)
    #     ee_action = self.ee_actor(ee_obs)

    #     # Combine actions and process through the final layer
    #     combined_action = th.cat([image_action, ee_action], dim=1)
    #     final_action = self.action_combiner(combined_action)

    #     if deterministic:
    #         return th.tanh(final_action)
    #     return final_action
    def forward(self, obs, deterministic=False):
        # Extract the image and other components from the observation
        image_obs = obs['image']  # Assuming 'image' is part of the observation dict

        # Convert the image to float type
        image_obs = image_obs.to(th.float32)  # Cast to float32

        # Pass the image through the CNN actor
        image_action = self.image_actor(image_obs)
        print(f"Shape before fully connected: {image_action.shape}")
        # image_action = image_action.view(image_action.size(0), -1)
        
        # Assuming 'end_effector_pos' is also part of the observation
        end_effector_pos = obs['end_effector_pos']
        ee_action = self.ee_actor(end_effector_pos)
        combined_action = th.cat([image_action, ee_action], dim=1)
        final_action = self.action_combiner(combined_action)

        if deterministic:
            return th.tanh(final_action)
        return final_action

    def _predict(self, obs, deterministic=False):
        return self.forward(obs, deterministic)

    def make_critic(self, features_extractor=None):
        """
        Create a critic network following Stable-Baselines3 SAC structure.
        """
        # Define the critic network for both Q-values
        return ContinuousCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=None,  # No additional features extractor
            features_dim=self.observation_space['end_effector_pos'].shape[0] + 
                        self.observation_space['image'].shape[0],
            net_arch=self.net_arch,  # Pass the same network architecture
            activation_fn=self.activation_fn,
            n_critics=2,  # Default: 2 Q-networks
        )
