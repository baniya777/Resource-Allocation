"""
independent Q-learning for police and thief agents
each agent has its own Q-table and learns from its own rewards
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import pandas as pd
import seaborn as sns
import networkx as nx


class MultiAgentQLearning:

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.env     = env
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min   = 0.01

        self.q_tables = {agent: {} for agent in env.possible_agents}

        # training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.response_times  = []
        self.success_rates   = []

        # FIX 3 (scalability): track Q-table size to monitor state explosion
        self.q_table_sizes = []

    def get_state_key(self, state_dict, active_agents=None):
        """
        FIX 2: Only include ACTIVE (not yet caught) thieves in the state key.
        Caught thieves are excluded entirely so that genuinely different
        game situations are not collapsed to the same key.

        FIX 3 (scalability): Use node INDEX (integer 0..N-1) instead of the
        full node-id string. This makes keys smaller and hashing faster.
        """
        node_idx = self.env.node_inv_dict   # maps node_id -> integer index

        police_pos = tuple(sorted(
            (a, node_idx[state_dict[a]])
            for a in self.env.possible_responders
            if state_dict.get(a) is not None
        ))

        # FIX 2: only include thieves that are still active
        active = active_agents if active_agents is not None else self.env.agents
        active_set = set(active)
        thief_pos = tuple(sorted(
            (a, node_idx[state_dict[a]])
            for a in self.env.possible_criminals
            if a in active_set and state_dict.get(a) is not None
        ))

        return (police_pos, thief_pos)

    def choose_action(self, agent, state_key, training=True):
        """
        FIX 1 (action space): sample only from the valid actions for the
        agent's current node, not blindly from 0..max_actions-1.
        """
        valid_n = self.env.get_valid_action_count(agent)

        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, valid_n)

        if state_key not in self.q_tables[agent]:
            self.q_tables[agent][state_key] = {}

        q_vals = self.q_tables[agent][state_key]
        if not q_vals:
            return np.random.randint(0, valid_n)

        # only consider actions that are valid at this node
        valid_q = {a: v for a, v in q_vals.items() if a < valid_n}
        if not valid_q:
            return np.random.randint(0, valid_n)

        return max(valid_q, key=valid_q.get)

    def update_q_value(self, agent, state_key, action, reward, next_state_key):
        if state_key not in self.q_tables[agent]:
            self.q_tables[agent][state_key] = {}
        if next_state_key not in self.q_tables[agent]:
            self.q_tables[agent][next_state_key] = {}

        current_q  = self.q_tables[agent][state_key].get(action, 0.0)
        next_qs    = self.q_tables[agent][next_state_key]
        max_next_q = max(next_qs.values()) if next_qs else 0.0

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_tables[agent][state_key][action] = new_q

    def train(self, num_episodes=1000, verbose=True):
        print(f"starting training: {len(self.env.possible_agents)} agents, {num_episodes} episodes")
        print(f"police: {self.env.possible_responders}")
        print(f"thieves: {self.env.possible_criminals}")
        print()

        for episode in range(num_episodes):
            state     = self.env.reset()
            # FIX 2: pass active agents so caught thieves are excluded from key
            state_key = self.get_state_key(state, self.env.agents)

            episode_reward = {a: 0.0 for a in self.env.possible_agents}
            step = 0
            done = False

            while not done and step < self.env.max_steps:
                actions = {
                    a: self.choose_action(a, state_key, training=True)
                    for a in self.env.agents
                }
                next_state, rewards, terminations, truncations, infos = \
                    self.env.step(actions)

                # FIX 2: pass updated active agent list
                next_state_key = self.get_state_key(next_state, self.env.agents)

                for agent in list(self.env.agents):
                    if agent in actions:
                        r = rewards.get(agent, 0.0)
                        self.update_q_value(
                            agent, state_key, actions[agent], r, next_state_key
                        )
                        episode_reward[agent] += r

                state_key = next_state_key
                step     += 1

                if any(terminations.values()):
                    done = True

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(sum(episode_reward.values()))
            self.episode_lengths.append(step)

            metrics = self.env.get_metrics()
            self.response_times.append(metrics["avg_response_time"])
            self.success_rates.append(metrics["success_rate"])

            # FIX 3: track total unique states visited across all agents
            total_states = sum(len(qt) for qt in self.q_tables.values())
            self.q_table_sizes.append(total_states)

            if verbose and (episode + 1) % 100 == 0:
                avg_r  = np.mean(self.episode_rewards[-100:])
                avg_l  = np.mean(self.episode_lengths[-100:])
                valid  = [r for r in self.response_times[-100:] if r > 0]
                avg_rt = np.mean(valid) if valid else 0.0
                avg_sr = np.mean(self.success_rates[-100:])
                print(f"ep {episode+1}/{num_episodes}  reward={avg_r:.1f}  "
                      f"len={avg_l:.0f}  catch_rate={avg_sr:.2%}  "
                      f"eps={self.epsilon:.3f}  states={total_states}")

        print("\ntraining done")
        return self.episode_rewards, self.episode_lengths

    def evaluate(self, num_episodes=100, render=False):
        print(f"\nevaluating {num_episodes} episodes...")

        eval_rewards       = []
        eval_response_times = []
        eval_success_rates  = []

        for episode in range(num_episodes):
            state     = self.env.reset()
            state_key = self.get_state_key(state, self.env.agents)

            total_reward = 0.0
            step = 0
            done = False

            while not done and step < self.env.max_steps:
                actions = {
                    a: self.choose_action(a, state_key, training=False)
                    for a in self.env.agents
                }
                next_state, rewards, terminations, _, _ = self.env.step(actions)
                next_state_key = self.get_state_key(next_state, self.env.agents)

                total_reward += sum(rewards.values())
                state_key     = next_state_key
                step         += 1

                if render and episode == 0:
                    self.env.temp_render(episode)

                if any(terminations.values()):
                    done = True

            eval_rewards.append(total_reward)
            metrics = self.env.get_metrics()
            eval_response_times.append(metrics["avg_response_time"])
            eval_success_rates.append(metrics["success_rate"])

        valid = [r for r in eval_response_times if r > 0]
        print(f"avg reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
        print(f"avg steps to catch: {np.mean(valid):.2f}" if valid else "no catches recorded")
        print(f"catch rate: {np.mean(eval_success_rates):.2%}")

        return eval_rewards, eval_response_times, eval_success_rates

    def save_policy(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                "q_tables": dict(self.q_tables),
                "alpha":    self.alpha,
                "gamma":    self.gamma,
            }, f)
        print(f"policy saved -> {filename}")

    def load_policy(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.q_tables = data["q_tables"]
        self.alpha    = data["alpha"]
        self.gamma    = data["gamma"]
        print(f"policy loaded from {filename}")

    def plot_training_progress(self, save_path="results/training_progress.png"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        window = 50

        datasets = [
            (axes[0, 0], self.episode_rewards, "total reward per episode",    "reward"),
            (axes[0, 1], self.episode_lengths,  "episode length (steps)",      "steps"),
            (axes[1, 0], self.response_times,   "steps to first catch",        "steps"),
            (axes[1, 1], self.success_rates,    "catch rate",                  "rate"),
            # FIX 3: extra panel showing Q-table growth over time
            (axes[0, 2], self.q_table_sizes,    "unique states visited (Q-table size)", "states"),
        ]

        for ax, data, title, ylabel in datasets:
            series = pd.Series(data)
            ax.plot(data, alpha=0.2, color="#636e72")
            if len(data) >= window:
                ax.plot(series.rolling(window).mean(), lw=2,
                        color="#2ecc71", label=f"{window}-ep moving avg")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("episode")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if ylabel == "rate":
                ax.set_ylim(0, 1)

        # hide unused subplot
        axes[1, 2].axis("off")

        fig.suptitle("training convergence - police and thief on planar graph",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"training plot saved -> {save_path}")
        plt.close()


# alias for backward compatibility
TrueMultiAgentLearning = MultiAgentQLearning


class BaselineComparison:

    def __init__(self, env):
        self.env = env
        self.results = {
            m: {"rewards": [], "response_times": [], "success_rates": []}
            for m in ("marl", "greedy", "random", "static")
        }

    def _greedy_policy(self, state):
        # each police moves one step toward the nearest thief
        actions = {}
        thief_nodes = [
            state[t] for t in self.env.possible_criminals
            if state.get(t) is not None
        ]

        for police in self.env.possible_responders:
            if state.get(police) is None:
                continue
            best_action = 3
            best_dist   = float("inf")

            for thief_node in thief_nodes:
                try:
                    path = nx.shortest_path(self.env.g_env, state[police], thief_node)
                    dist = len(path) - 1
                    if dist < best_dist:
                        best_dist = dist
                        if len(path) > 1:
                            next_node = path[1]
                            nbrs = list(self.env.g_env.neighbors(state[police]))
                            best_action = nbrs.index(next_node) if next_node in nbrs else 0
                        else:
                            best_action = 3
                except nx.NetworkXNoPath:
                    pass

            actions[police] = best_action

        for thief in self.env.possible_criminals:
            if state.get(thief) is None:
                continue
            nbrs = list(self.env.g_env.neighbors(state[thief]))
            actions[thief] = np.random.randint(0, max(len(nbrs), 1) + 1)

        return actions

    def _random_policy(self, state):
        # FIX 1: use valid action count per node, not a fixed 4
        actions = {}
        for agent in self.env.agents:
            if state.get(agent) is None:
                continue
            valid_n = self.env.get_valid_action_count(agent)
            actions[agent] = np.random.randint(0, valid_n)
        return actions

    def _static_policy(self, state):
        return {a: 3 for a in self.env.agents if state.get(a) is not None}

    def _evaluate_policy(self, name, policy_fn, num_episodes=100):
        print(f"  running {name}...")
        for _ in range(num_episodes):
            state        = self.env.reset()
            total_reward = 0.0
            done         = False
            step         = 0

            while not done and step < self.env.max_steps:
                actions = policy_fn(state)
                next_state, rewards, terminations, _, _ = self.env.step(actions)
                total_reward += sum(rewards.values())
                state         = next_state
                step         += 1
                if any(terminations.values()):
                    done = True

            metrics = self.env.get_metrics()
            self.results[name]["rewards"].append(total_reward)
            self.results[name]["response_times"].append(metrics["avg_response_time"])
            self.results[name]["success_rates"].append(metrics["success_rate"])

    def compare_all(self, marl_learner, num_episodes=100):
        print("running comparisons...")
        (self.results["marl"]["rewards"],
         self.results["marl"]["response_times"],
         self.results["marl"]["success_rates"]) = \
            marl_learner.evaluate(num_episodes=num_episodes)

        self._evaluate_policy("greedy", self._greedy_policy, num_episodes)
        self._evaluate_policy("random", self._random_policy, num_episodes)
        self._evaluate_policy("static", self._static_policy, num_episodes)

    def generate_report(self):
        print("\n" + "-" * 55)
        print("baseline comparison results")
        print("-" * 55)
        for method, data in self.results.items():
            if not data["rewards"]:
                continue
            valid = [r for r in data["response_times"] if r > 0]
            rt_str = f"{np.mean(valid):.2f}" if valid else "n/a"
            print(f"\n{method}")
            print(f"  avg reward:    {np.mean(data['rewards']):.2f} +/- {np.std(data['rewards']):.2f}")
            print(f"  steps to catch: {rt_str}")
            print(f"  catch rate:    {np.mean(data['success_rates']):.2%}")

    def plot_comparison(self, save_path="results/baseline_comparison.png"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        methods = ["marl", "greedy", "random", "static"]
        colors  = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
        labels  = [m.upper() for m in methods]

        configs = [
            (axes[0], "rewards",        "total reward",    "reward comparison"),
            (axes[1], "response_times", "steps to catch",  "catch speed"),
            (axes[2], "success_rates",  "catch rate",      "success rate"),
        ]

        for ax, key, ylabel, title in configs:
            data = [
                ([r for r in self.results[m][key] if r > 0]
                 if key == "response_times" else self.results[m][key])
                for m in methods
            ]
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if key == "success_rates":
                ax.set_ylim(0, 1)

        fig.suptitle("MARL vs baselines - police and thief resource allocation",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"comparison plot saved -> {save_path}")
        plt.close()


if __name__ == "__main__":
    print("MultiAgentAlgorithm loaded")
