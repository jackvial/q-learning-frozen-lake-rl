import numpy as np


class QLearningAgent:
    def __init__(self, env) -> None:
        self.env = env
        self.q_table = self.build_q_table(env.observation_space.n, env.action_space.n)

    def build_q_table(self, n_states, n_actions):
        return np.zeros((n_states, n_actions))

    def epsilon_greedy_policy(self, state, epsilon):

        # Epsilon probability of taking a random action or the
        # action that has the highest Q value for the current state
        if np.random.random() < epsilon:
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.q_table[state])

    def greedy_policy(self, state):
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, gamma, learning_rate, new_state):

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[new_state])
        self.q_table[state][action] = current_q + learning_rate * (
            reward + gamma * next_max_q - current_q
        )
