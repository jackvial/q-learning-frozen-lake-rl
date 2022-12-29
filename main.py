import time
import curses
import numpy as np
import warnings
from environment import FrozenLakeEnvCustom
from agent import QLearningAgent

warnings.filterwarnings("ignore")


def train_agent(
    n_training_episodes,
    min_epsilon,
    max_epsilon,
    decay_rate,
    env,
    max_steps,
    agent,
    learning_rate,
    gamma,
    use_frame_delay,
):
    for episode in range(n_training_episodes + 1):

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        state, info = env.reset()
        done = False
        for step in range(max_steps):

            # Choose the action At using epsilon greedy policy
            action = agent.epsilon_greedy_policy(state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, gamma, learning_rate, new_state)

            env.render(
                title=f"Training: {episode}/{n_training_episodes}",
                q_table=agent.q_table,
            )

            if use_frame_delay:
                time.sleep(0.01)

            if done:
                break

            state = new_state
    return agent


def evaluate_agent(env, max_steps, n_eval_episodes, agent, seed, use_frame_delay):
    successful_episodes = []
    episodes_slips = []
    for episode in range(n_eval_episodes + 1):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        done = False
        total_rewards_ep = 0

        slips = []
        for step in range(max_steps):

            # Take the action (index) that have the maximum expected future reward given that state
            action = agent.greedy_policy(state)

            expected_new_state = env.get_expected_new_state_for_action(action)
            new_state, reward, done, truncated, info = env.step(action)
            total_rewards_ep += reward

            if expected_new_state != new_state:
                slips.append((step, action, expected_new_state, new_state))

            if reward != 0:
                successful_episodes.append(episode)

            env.render(
                title=f"Evaluating: {episode}/{n_eval_episodes} | Slips: {len(slips)}",
                q_table=agent.q_table,
            )
            episodes_slips.append(len(slips))

            if use_frame_delay:
                time.sleep(0.01)

            if done:
                break
            state = new_state

    mean_slips = np.mean(episodes_slips)
    return successful_episodes, mean_slips


def main(screen):

    # Training parameters
    n_training_episodes = 2000  # Total training episodes
    learning_rate = 0.1  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100  # Total number of test episodes

    # Environment parameters
    max_steps = 99  # Max steps per episode
    gamma = 0.99  # Discounting rate
    eval_seed = []  # The evaluation seed of the environment

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.0005  # Exponential decay rate for exploration prob

    use_frame_delay = False

    env = FrozenLakeEnvCustom(map_name="4x4", is_slippery=True, render_mode="curses")

    agent = QLearningAgent(env)
    agent = train_agent(
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        env,
        max_steps,
        agent,
        learning_rate,
        gamma,
        use_frame_delay,
    )

    successful_episodes, mean_slips = evaluate_agent(
        env, max_steps, n_eval_episodes, agent, eval_seed, use_frame_delay
    )

    env_curses_screen = env.curses_screen
    env_curses_screen.addstr(
        5,
        2,
        f"Successful episodes: {len(successful_episodes)}/{n_eval_episodes} | Avg slips: {mean_slips:.2f}\n\n",
    )
    env_curses_screen.noutrefresh()
    curses.doupdate()
    time.sleep(10)


if __name__ == "__main__":

    # Reset the terminal state after using curses
    # Call main("") instead if you want to leave the final state of the environment
    # on the terminal
    curses.wrapper(main)
