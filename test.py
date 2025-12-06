import torch
import gymnasium as gym
import flappy_bird_gymnasium
from model import A2C
import os
import time

def load_best_model(checkpoint_path, state_dim, action_dim, device):
    """Load the best saved model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Initialize model
    model = A2C(state_dim, action_dim, device=str(device))
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.actor.load_state_dict(checkpoint['actor_state_dict'])
    model.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    print(f"Model loaded from episode {checkpoint['episode']}")
    print(f"Best average reward: {checkpoint['best_avg_reward']:.1f}")
    
    return model

def test_agent(model, env, device, num_episodes=5):
    """Test the agent for multiple episodes"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device)
        episode_reward = 0
        step = 0
        
        done = False
        while not done:
            # Select action using the trained policy (no exploration)
            with torch.no_grad():
                action_probs = model.actor(state)
            action = torch.argmax(action_probs).item()
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = torch.from_numpy(next_state).float().to(device)
            step += 1
            
            # Small delay for visualization
            # time.sleep(0.01)
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.1f} | Steps: {step} | Score: {info['score']}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.1f}")
    return total_rewards

if __name__ == "__main__":
    # Setup
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create environment with human render mode for visualization
    env = gym.make('FlappyBird-v0', render_mode= 'human', use_lidar= False)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Load the best model
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'best_model.pth')
    model = load_best_model(checkpoint_path, state_dim, action_dim, device)
    
    # Test the agent
    print("\nStarting test episodes...\n")
    test_agent(model, env, device, num_episodes=1)
    
    env.close()
    print("\nTesting complete!")
