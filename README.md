"# A2C Flappy Bird - Reinforcement Learning

An implementation of the Advantage Actor-Critic (A2C) algorithm for training an agent to play Flappy Bird using PyTorch.

## Features

- 🎮 A2C (Advantage Actor-Critic) algorithm implementation
- 🏆 Best model saving based on moving average reward (100 episodes)
- 📊 TensorBoard integration for training visualization
- 💾 Automatic checkpoint saving and resume training capability
- 🎯 Testing script with visual rendering
- ⚡ Gradient clipping and entropy regularization

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- TensorBoard

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/ntda1972000/reinforcement-learning-A2C.git
cd reinforcement-learning-A2C
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── train.py          # Main training script
├── test.py           # Testing script with visualization
├── model.py          # A2C model architecture
├── buffer.py         # Replay buffer implementation
├── parameters.py     # Hyperparameters configuration
├── requirements.txt  # Python dependencies
├── checkpoints/      # Saved models (created during training)
└── runs/             # TensorBoard logs (created during training)
```

## Usage

### Training

Start training the agent:

```bash
python train.py
```

**Training Features:**
- Automatically loads the best saved model if available (resume training)
- Saves the best model based on 100-episode moving average reward
- Creates periodic checkpoints every 200 episodes
- Logs training metrics to TensorBoard

**View Training Progress:**

In a separate terminal, run:
```bash
tensorboard --logdir=runs
```
Then open your browser and navigate to `http://localhost:6006`

### Testing

Test the trained agent with visual rendering:

```bash
python test.py
```

This will:
- Load the best saved model from `checkpoints/best_model.pth`
- Run 5 test episodes with visualization
- Display performance statistics

### Configuration

Edit `parameters.py` to customize hyperparameters:

```python
# Training parameters
critic_lr = 5e-4           # Critic learning rate
actor_lr = 1e-4            # Actor learning rate
gamma = 0.99               # Discount factor
max_episode = 50_000       # Maximum training episodes
max_step = 1e5             # Maximum steps per episode

# A2C specific
entropy_coef = 0.03        # Entropy coefficient
value_coef = 0.5           # Value loss coefficient
max_grad_norm = 0.5        # Gradient clipping threshold

# Exploration
epsilon = 0.0              # Initial exploration rate
epsilon_decay = 0.97       # Epsilon decay rate
epsilon_min = 0.0          # Minimum epsilon
```

## Model Architecture

### Actor Network
- Input: State features (12 dimensions)
- Hidden layers: 128 → 128 neurons with ReLU activation
- Output: Action probabilities (Softmax)

### Critic Network
- Input: State features (12 dimensions)
- Hidden layers: 64 → 64 neurons with ReLU activation
- Output: State value estimation

## Training Details

- **Algorithm**: Advantage Actor-Critic (A2C)
- **Policy**: Stochastic policy with Categorical distribution
- **Advantage Estimation**: TD(0) temporal difference
- **Optimization**: Separate Adam optimizers for actor and critic
- **Loss Functions**:
  - Actor: Policy gradient with entropy regularization
  - Critic: Mean Squared Error (MSE) for value prediction

## Saved Models

Models are saved in the `checkpoints/` directory:

- `best_model.pth`: Best model based on 100-episode moving average reward
- `checkpoint_model.pth`: Periodic checkpoint (every 200 episodes)

Each checkpoint contains:
- Actor and Critic network weights
- Optimizer states
- Episode number
- Best rewards
- Epsilon value

## Tips for Better Training

1. **Monitor TensorBoard**: Watch the moving average reward trend
2. **Adjust hyperparameters**: Tune learning rates if training is unstable
3. **Training time**: Let it train for several thousand episodes for good results
4. **Resume training**: Simply run `train.py` again to continue from the best checkpoint

## Troubleshooting

**CUDA Issues:**
The code is configured to use CPU by default. To enable GPU:
```python
# In train.py and test.py, change:
device = torch.device('cpu')
# to:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Module not found errors:**
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## License

MIT License

## Author

ntda1972000

## Acknowledgments

- Flappy Bird Gymnasium environment
- PyTorch team for the deep learning framework
- OpenAI Gymnasium for the RL interface
" 
