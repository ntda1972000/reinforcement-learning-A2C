import torch
import random
from model import A2C
from buffer import ReplayBuffer
import gymnasium as gym
import flappy_bird_gymnasium
import os
import torch.optim as optim
import torch.nn as nn
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from parameters import (
    critic_lr,
    actor_lr,
    gamma,
    max_episode,
    max_step,
    buffer_len,
    entropy_coef,
    value_coef,
    max_grad_norm,
    batch_size,
    rollout_len,
    epsilon,
    epsilon_decay,
    epsilon_min
)

def choose_device():
    if torch.cuda.is_available():
        device_name = 'cuda'
    else:
        device_name = 'cpu'
    return torch.device(device_name)

def set_learning_rate(optimizer, new_learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_learning_rate

def load_check_point(checkpoint_path, online_net, target_net, actor_optimizer, critic_optimizer):
    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            online_net.actor.load_state_dict(checkpoint['actor_state_dict'])
            online_net.critic.load_state_dict(checkpoint['critic_state_dict'])
            actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            target_net.actor.load_state_dict(online_net.actor.state_dict())
            target_net.critic.load_state_dict(online_net.critic.state_dict())
            
            episode_start = checkpoint.get('episode', 0) + 1
            best_reward = checkpoint.get('best_reward', float('-inf'))
            best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
            epsilon = checkpoint.get('epsilon', 0.0)
            
            print(f'Checkpoint loaded from episode {checkpoint["episode"]}')
            print(f'Best reward: {best_reward:.1f}')
            print(f'Best avg reward: {best_avg_reward:.1f}')
            
            return episode_start, best_reward, best_avg_reward, epsilon
    except Exception as e:
        print(f'No checkpoint exists or error loading: {checkpoint_path}')
        print(f'Error: {e}')
    
    return 0, float('-inf'), float('-inf'), epsilon

def entropy(value):
    return -(value * torch.log(value + 1e-8)).sum()

env =gym.make('FlappyBird-v0', render_mode = None, use_lidar = False)
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

replay_buffer = ReplayBuffer(buffer_len)

# Force CPU due to CUDA compatibility issues
device = torch.device('cpu')
print(f"Using device: {device}")
online_net = A2C(state_dim, action_dim, device=str(device))
critic_optimizer = optim.Adam(online_net.critic.parameters(), lr = critic_lr)
actor_optimizer = optim.Adam(online_net.actor.parameters(), lr = actor_lr)
target_net = A2C(state_dim, action_dim, device=str(device))
# Initialize target network with online network weights
target_net.actor.load_state_dict(online_net.actor.state_dict())
target_net.critic.load_state_dict(online_net.critic.state_dict())

# Try to load best model if it exists
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

episode_start, best_reward, best_avg_reward, loaded_epsilon = load_check_point(
    best_model_path, online_net, target_net, actor_optimizer, critic_optimizer
)

# Use loaded epsilon if available, otherwise use default from parameters
from parameters import epsilon as default_epsilon
epsilon = loaded_epsilon if episode_start > 0 else default_epsilon

global_step = 0
reward_window = deque(maxlen=100)  # Moving average over last 100 episodes
writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), 'runs', 'a2c_flappy'))

for episode in range(0, int(max_episode)):
    state, _ = env.reset()
    state = torch.from_numpy(state).float().to(device)
    episode_reward = 0
    critic_loss = torch.tensor(0.0)
    actor_loss = torch.tensor(0.0)
    
    # Collect trajectory for this episode
    states_list, actions_list, rewards_list, dones_list = [], [], [], []
    
    for step in range(int(max_step)):
        with torch.no_grad():
            action_probs = online_net.actor(state)
        # Choose an action based on action prob
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()

        # Perform the chosen action to get the next state
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.from_numpy(next_state).float().to(device)
        
        # Store trajectory
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        
        episode_reward += reward
        state = next_state
        global_step += 1
        
        if done:
            break
    
    # Update every episode (on-policy)
    if len(states_list) > 0:
        states_t = torch.stack(states_list)
        actions_t = torch.tensor(actions_list, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones_list, dtype=torch.float32, device=device)
        
        # Compute values and next values
        values = online_net.critic(states_t).squeeze(-1)
        with torch.no_grad():
            next_values = torch.zeros_like(rewards_t)
            next_values[:-1] = online_net.critic(states_t[1:]).squeeze(-1)
            td_targets = rewards_t + gamma * next_values * (1.0 - dones_t)
            advantages = td_targets - values.detach()
        
        # Actor loss
        actor_optimizer.zero_grad()
        action_probs = online_net.actor(states_t)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(log_probs * advantages).mean() - entropy_coef * dist.entropy().mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.actor.parameters(), max_grad_norm)
        actor_optimizer.step()
        
        # Critic loss
        critic_optimizer.zero_grad()
        critic_loss = torch.nn.functional.mse_loss(values, td_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.critic.parameters(), max_grad_norm)
        critic_optimizer.step()
    
    epsilon = max(epsilon*epsilon_decay, epsilon_min)
    
    # Update reward window for moving average
    reward_window.append(episode_reward)
    avg_reward = sum(reward_window) / len(reward_window)
    
    # Save best model based on moving average
    if avg_reward > best_avg_reward and len(reward_window) >= 10:  # Wait for at least 10 episodes
        best_avg_reward = avg_reward
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save({
            'episode': episode,
            'actor_state_dict': online_net.actor.state_dict(),
            'critic_state_dict': online_net.critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'best_avg_reward': best_avg_reward,
            'avg_reward': avg_reward,
            'epsilon': epsilon
        }, best_model_path)
        print(f'\nNew best model saved! Episode: {episode + 1}, Avg Reward (100 ep): {avg_reward:.1f}')
    
    # Track best single episode reward
    if episode_reward > best_reward:
        best_reward = episode_reward
    
    if episode % 200 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_model.pth')
        torch.save({
            'episode': episode,
            'actor_state_dict': online_net.actor.state_dict(),
            'critic_state_dict': online_net.critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'best_reward': best_reward,
            'best_avg_reward': best_avg_reward,
            'epsilon': epsilon
        }, checkpoint_path)
    # TensorBoard logging per episode
    writer.add_scalar('loss/critic', critic_loss.item(), episode)
    writer.add_scalar('loss/actor', actor_loss.item(), episode)
    writer.add_scalar('train/episode_reward', episode_reward, episode)
    writer.add_scalar('train/avg_reward', avg_reward, episode)
    writer.add_scalar('train/best_reward', best_reward, episode)
    writer.add_scalar('train/best_avg_reward', best_avg_reward, episode)
    print(f'episode {episode + 1}/{int(max_episode)} |Critic Loss: {critic_loss.item():.3f} |Actor Loss: {actor_loss.item():.3f} |Reward: {episode_reward:.1f} |Avg: {avg_reward:.1f} | Step: {step}'  + 6*' ', end = '\r')

# Close writer when training ends
writer.close()