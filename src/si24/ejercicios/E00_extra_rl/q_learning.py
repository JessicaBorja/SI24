import gymnasium as gym
import numpy as np

class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

        # Tabla estados x acciones
        self.Q = np.zeros((env.observation_space.n,
                           env.action_space.n))
        # Parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="human")

    n_episodes = 1000
    episode_length = 200
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.9)
    for e in range(n_episodes):
        obs, _ = env.reset()  # Corregir aquí, desempaquetar la tupla
        ep_return = 0
        for i in range(episode_length):
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.step(obs, action, reward, next_obs)

            if done:
                break
            ep_return += reward
            obs = next_obs
            #env.render()

        # Reducir la exploración del agente conforme aprende
        agent.epsilon *= 0.8  # Reducción lineal de epsilon

        print(f"Episode {e} return: ", ep_return)
    env.close()
