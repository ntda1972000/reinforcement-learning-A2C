# Training parameter
critic_lr = 1e-4
actor_lr = 1e-4
gamma = 0.99
max_episode = 5_000
max_step = 10_000
buffer_len = 2_000

# Actor Critic specifics
entropy_coef = 0.03
value_coef = 0.5
max_grad_norm = 0.6

# Optimization and rollout
batch_size = 64
rollout_len = 20
gae_lambda = 0.95

# Epsilon greedy
epsilon = 0.0
epsilon_decay = 0.97
epsilon_min = 0.0
