"""
full training pipeline for MARL resource allocation
police and thief scenario on a planar graph
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from EmergencyResponseEnv import EmergencyResponseEnvironment
from MultiAgentAlgorithm import TrueMultiAgentLearning, BaselineComparison
from VideoGenerator import VideoGenerator


print("resource allocation - multi agent reinforcement learning")
print("police and thief on planar graph")
print()

# ---  environment  -----------------------------------------------------------

print("setting up environment...")

env = EmergencyResponseEnvironment(
    graph_path="g1.graphml",   # FIX 4: correct .graphml extension
    num_police=5,
    num_thieves=2,
)

print(f"  nodes: {env.g_no_node}")
print(f"  police: {env.no_of_responders},  thieves: {env.no_of_criminals}")
print(f"  max actions per agent: {env.max_actions}")
print()

# ---  learner  ---------------------------------------------------------------

print("initialising Q-learner...")

learner = TrueMultiAgentLearning(
    env=env,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.3,
)

# ---  training  --------------------------------------------------------------

num_episodes = 2000

print(f"training for {num_episodes} episodes...")
print()

episode_rewards, episode_lengths = learner.train(
    num_episodes=num_episodes,
    verbose=True,
)

print()

# ---  save  ------------------------------------------------------------------

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("images",  exist_ok=True)

learner.save_policy("models/trained_policy.pkl")
print()

# ---  evaluate  --------------------------------------------------------------

print("evaluating trained policy...")

eval_rewards, eval_response_times, eval_success_rates = learner.evaluate(
    num_episodes=100,
    render=False,
)
print()

# ---  baselines  -------------------------------------------------------------

print("running baseline comparisons...")
print()

comparator = BaselineComparison(env)
comparator.compare_all(learner, num_episodes=100)
print()
comparator.generate_report()

# ---  plots  -----------------------------------------------------------------

print()
learner.plot_training_progress(save_path="results/training_progress.png")
comparator.plot_comparison(save_path="results/baseline_comparison.png")

# ---  final report  ----------------------------------------------------------

def safe_pct(a, b):
    if b == 0:
        return float("nan")
    return (a / b - 1) * 100

def avg_rt(key):
    valid = [r for r in comparator.results[key]["response_times"] if r > 0]
    return np.mean(valid) if valid else 0.0

marl_r   = np.mean(comparator.results["marl"]["rewards"])
greedy_r = np.mean(comparator.results["greedy"]["rewards"])
random_r = np.mean(comparator.results["random"]["rewards"])
static_r = np.mean(comparator.results["static"]["rewards"])

report = f"""
{'='*60}
FINAL REPORT
{'='*60}

config:
  episodes  : {num_episodes}
  nodes     : {env.g_no_node}
  police    : {env.no_of_responders}
  thieves   : {env.no_of_criminals}
  max actions per node : {env.max_actions}
  alpha/gamma/epsilon_final: {learner.alpha}/{learner.gamma}/{learner.epsilon:.3f}

MARL results (100 eval episodes):
  avg reward     : {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}
  steps to catch : {avg_rt('marl'):.2f}
  catch rate     : {np.mean(eval_success_rates):.2%}

comparison:
  method    reward    steps_to_catch   catch_rate
  marl      {marl_r:>8.2f}   {avg_rt('marl'):>14.2f}   {np.mean(comparator.results['marl']['success_rates']):>9.1%}
  greedy    {greedy_r:>8.2f}   {avg_rt('greedy'):>14.2f}   {np.mean(comparator.results['greedy']['success_rates']):>9.1%}
  random    {random_r:>8.2f}   {avg_rt('random'):>14.2f}   {np.mean(comparator.results['random']['success_rates']):>9.1%}
  static    {static_r:>8.2f}   {avg_rt('static'):>14.2f}   {np.mean(comparator.results['static']['success_rates']):>9.1%}

improvement over baselines:
  vs greedy : {safe_pct(marl_r, greedy_r):>+.1f}%
  vs random : {safe_pct(marl_r, random_r):>+.1f}%
  vs static : {safe_pct(marl_r, static_r):>+.1f}%
{'='*60}
"""

print(report)
with open("results/final_report.txt", "w") as f:
    f.write(report)

# ---  demo episode  ----------------------------------------------------------

print("running one demo episode with trained policy")
print()

state     = env.reset()
state_key = learner.get_state_key(state, env.agents)

for step in range(30):
    actions = {
        a: learner.choose_action(a, state_key, training=False)
        for a in env.agents
    }
    next_state, rewards, terminations, truncations, infos = env.step(actions)
    next_state_key = learner.get_state_key(next_state, env.agents)

    caught = infos["global"]["resolved_this_step"]
    if caught:
        print(f"step {step+1:>2}: caught {caught}")
    else:
        print(f"step {step+1:>2}: running  (total catches: {env.catches})")

    state_key = next_state_key
    if any(terminations.values()):
        print("episode ended")
        break

print()
print("training complete - results in results/")
print()

# ---  videos  ----------------------------------------------------------------

print("generating videos...")
print()

video_gen = VideoGenerator(env, learner, comparator, fps=10, dpi=120)
video_gen.generate_all_videos()

print()
print("videos saved to videos/:")
print("  training_process.mp4    - full training replay")
print("  trained_agents_demo.mp4 - trained agents in action")
print("  police_catches_thief.mp4 - pursuit scenario")
print("  baseline_comparison.mp4 - MARL vs baselines")
