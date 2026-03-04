"""
police and thief environment for MARL resource allocation
runs on a planar graph loaded from graphml file
"""

from pettingzoo.utils.env import ParallelEnv
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt
import random


class PoliceThiefEnvironment(ParallelEnv):
    # two agent types: police try to catch thieves, thieves try to escape
    # graph is loaded from graphml, agents move along edges

    metadata = {"name": "police_thief_v1"}

    def __init__(self, graph_path="g1.graphml", num_police=5, num_thieves=2):
        self.g_env = nx.read_graphml(graph_path)
        self.node_list = list(self.g_env.nodes())
        self.g_no_node = len(self.node_list)

        self.node_dict = {i: n for i, n in enumerate(self.node_list)}
        self.node_inv_dict = {n: i for i, n in self.node_dict.items()}

        self.no_of_responders = num_police
        self.no_of_criminals = num_thieves

        self.possible_responders = [f"police_{i}" for i in range(num_police)]
        self.possible_criminals  = [f"thief_{i}"  for i in range(num_thieves)]
        self.possible_agents     = self.possible_responders + self.possible_criminals

        # FIX 1: action space is now per-node (dynamic), but we store the MAX
        # degree so gymnasium has a fixed space. Actual valid actions are
        # resolved at step-time using the real neighbour list.
        max_degree = max(d for _, d in self.g_env.degree())
        # +1 for the "stay" action
        self.max_actions = max_degree + 1
        self._action_spaces = {
            a: Discrete(self.max_actions) for a in self.possible_agents
        }
        self._observation_spaces = {
            a: Discrete(self.g_no_node) for a in self.possible_agents
        }
        self.action_spaces      = self._action_spaces
        self.observation_spaces = self._observation_spaces

        # precompute per-node action counts so choose_action can sample correctly
        self.node_action_count = {
            node: self.g_env.degree(node) + 1   # neighbours + stay
            for node in self.node_list
        }

        # get node positions from node id strings like "(0.23, 0.45)"
        self.node_positions = {}
        for node in self.node_list:
            try:
                x, y = self._str_to_tuple(node)
                self.node_positions[node] = (x, y)
            except Exception:
                pass
        if not self.node_positions:
            pos = nx.spring_layout(self.g_env, seed=42)
            self.node_positions = {n: tuple(pos[n]) for n in self.node_list}

        self.max_steps = 200
        self.state        = {}
        self.agents       = list(self.possible_agents)
        self.terminations = {a: False for a in self.possible_agents}
        self.current_time = 0

        self.catches              = 0
        self.steps_to_first_catch = None

        # kept for compatibility with training script
        self.events_resolved   = 0
        self.events_failed     = 0
        self.active_emergencies = []

        self._initialize_positions()

    def _str_to_tuple(self, s):
        return tuple(float(v) for v in s.strip("()").split(","))

    def _initialize_positions(self):
        for agent in self.possible_agents:
            self.state[agent] = random.choice(self.node_list)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_valid_action_count(self, agent):
        """Returns the real number of valid actions for the agent's current node."""
        return self.node_action_count[self.state[agent]]

    def reset(self, seed=None, options=None):
        self.agents       = list(self.possible_agents)
        self.terminations = {a: False for a in self.possible_agents}
        self.current_time         = 0
        self.catches              = 0
        self.steps_to_first_catch = None
        self.events_resolved      = 0
        self.events_failed        = 0
        self.active_emergencies   = []
        self._initialize_positions()
        return dict(self.state)

    def step(self, actions):
        self.current_time += 1

        rewards      = {a: 0.0   for a in self.agents}
        terminations = {a: False  for a in self.agents}
        truncations  = {a: False  for a in self.agents}
        infos        = {a: {}     for a in self.agents}

        for agent in self.agents:
            if agent not in actions:
                continue
            neighbours = list(self.g_env.neighbors(self.state[agent]))
            act = actions[agent]
            # FIX 1: last valid action index = len(neighbours), meaning "stay"
            # Any action index < len(neighbours) moves to that neighbour
            if act < len(neighbours):
                self.state[agent] = neighbours[act]
                rewards[agent]   -= 1.0
            else:
                # stay — penalise slightly less than moving
                rewards[agent] -= 0.5

        caught_thieves = set()
        for thief in self.possible_criminals:
            if thief not in self.agents:
                continue
            for police in self.possible_responders:
                if police not in self.agents:
                    continue
                if self.state[police] == self.state[thief]:
                    rewards[police]      += 200.0
                    rewards[thief]       -= 200.0
                    terminations[thief]   = True
                    caught_thieves.add(thief)
                    self.catches         += 1
                    self.events_resolved += 1
                    if self.steps_to_first_catch is None:
                        self.steps_to_first_catch = self.current_time
                    break

        for thief in self.possible_criminals:
            if thief not in caught_thieves and thief in self.agents:
                rewards[thief] += 1.0

        all_caught = all(terminations.get(t, False) for t in self.possible_criminals)
        timeout    = self.current_time >= self.max_steps

        if all_caught or timeout:
            if timeout and not all_caught:
                escaped = sum(
                    1 for t in self.possible_criminals
                    if not terminations.get(t, False)
                )
                for police in self.possible_responders:
                    rewards[police] -= escaped * 50.0
                self.events_failed += escaped
            for agent in self.agents:
                terminations[agent] = True

        infos["global"] = {
            "catches":              self.catches,
            "steps_to_first_catch": self.steps_to_first_catch,
            "resolved_this_step":   list(caught_thieves),
        }

        observations = {a: self.state[a] for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def temp_render(self, episode):
        import os
        os.makedirs("images", exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#0d0d1a")

        for u, v in self.g_env.edges():
            x0, y0 = self.node_positions[u]
            x1, y1 = self.node_positions[v]
            ax.plot([x0, x1], [y0, y1], color="#636e72", alpha=0.5, lw=1.2)

        xs = [self.node_positions[n][0] for n in self.node_list]
        ys = [self.node_positions[n][1] for n in self.node_list]
        ax.scatter(xs, ys, s=150, c="#636e72", edgecolors="#b2bec3",
                   linewidths=1, zorder=2, alpha=0.85)

        for p in self.possible_responders:
            if p in self.state and self.state[p]:
                x, y = self.node_positions[self.state[p]]
                ax.scatter(x, y, s=300, c="#74b9ff", marker="o",
                           edgecolors="white", linewidths=2, zorder=5)

        for t in self.possible_criminals:
            if t in self.state and self.state[t]:
                x, y = self.node_positions[self.state[t]]
                ax.scatter(x, y, s=300, c="#ff6b81", marker="X",
                           edgecolors="white", linewidths=2, zorder=5)

        ax.scatter([], [], c="#74b9ff", s=150, marker="o", label="Police")
        ax.scatter([], [], c="#ff6b81", s=150, marker="X", label="Thief")
        ax.legend(loc="upper right", facecolor="#12122a", labelcolor="white", fontsize=9)
        ax.set_title(
            f"Episode {episode}  step {self.current_time}  catches: {self.catches}",
            color="white", fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"images/police_thief_{episode}_{self.current_time}.png",
            dpi=150, bbox_inches="tight", facecolor="#0d0d1a",
        )
        plt.close()

    def get_metrics(self):
        total        = self.events_resolved + self.events_failed
        success_rate = self.events_resolved / total if total > 0 else 0.0
        avg_response = self.steps_to_first_catch if self.steps_to_first_catch else 0
        return {
            "avg_response_time": avg_response,
            "success_rate":      success_rate,
            "events_resolved":   self.events_resolved,
            "events_failed":     self.events_failed,
        }


EmergencyResponseEnvironment = PoliceThiefEnvironment


if __name__ == "__main__":
    print("police thief env loaded")
