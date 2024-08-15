import torch
from custompolicytest import PPO

# Training the PPO Agent
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")  # render_mode="human"
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ppo_agent = PPO(state_dim, action_dim)

ppo_path = "/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/ppo_agent_2.pth"
ppo_agent.policy.load_state_dict(torch.load(ppo_path))
ppo_agent.policy.eval()

state, _ = env.reset()

total_reward = 0
done = False

while not done:
    env.render()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_probs, _ = ppo_policy(state)
    action = torch.argmax(action_probs).item()
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f'Total Reward: {total_reward}')
env.close()