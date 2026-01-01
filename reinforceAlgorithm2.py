import gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Hyperparameters (Section 4.4)
# -----------------------------
LEARNING_RATE = 0.001
GAMMA = 0.99
MAX_EPISODES = 3000

# -----------------------------
# 2. Utility Functions
# -----------------------------

def softmax(x):
    """Compute softmax values for vector x."""
    x = x - np.max(x)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def discount_rewards(rewards, gamma):
    """Compute discounted returns G_t."""
    discounted = np.zeros_like(rewards, dtype=np.float64)
    running_sum = 0.0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    return discounted

# -----------------------------
# 3. Policy Representation (Section 4.2)
# -----------------------------

class Policy:
    def __init__(self, state_dim, action_dim):
        # Linear policy parameters
        self.theta = np.random.randn(state_dim, action_dim) * 0.01

    def action_probs(self, state):
        """π(a|s) using softmax"""
        z = np.dot(state, self.theta)
        return softmax(z)

    def sample_action(self, state):
        """Sample action from policy"""
        probs = self.action_probs(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def update(self, states, actions, returns, lr):
        """Policy gradient update"""
        for s, a, G in zip(states, actions, returns):
            probs = self.action_probs(s)
            grad_log = -probs
            grad_log[a] += 1
            self.theta += lr * G * np.outer(s, grad_log)

# -----------------------------
# 4. Training Loop (Section 4.3)
# -----------------------------

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = Policy(state_dim, action_dim)

episode_rewards = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    states, actions, rewards = [], [], []

    done = False
    while not done:
        action, _ = policy.sample_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    # Compute returns (Monte Carlo)
    returns = discount_rewards(rewards, GAMMA)

    # Normalize returns (optional but improves stability)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    # Update policy
    policy.update(states, actions, returns, LEARNING_RATE)

    total_reward = sum(rewards)
    episode_rewards.append(total_reward)

    # Logging
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}, Average Reward (last 100): {avg_reward:.2f}")

env.close()

# -----------------------------
# 5. Visual Results (Section 5.3)
# -----------------------------

episodes = np.arange(1, len(episode_rewards) + 1)
window = 50
moving_avg = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure()
plt.plot(episodes, episode_rewards, alpha=0.4, label="Episode Reward")
plt.plot(
    episodes[window - 1:],
    moving_avg,
    linewidth=2,
    label="Moving Average (50 episodes)"
)

plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("REINFORCE Performance on CartPole-v1")
plt.legend()
plt.grid(True)
plt.show()
