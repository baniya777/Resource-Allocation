"""
VIDEO GENERATOR FOR MARL EMERGENCY RESPONSE
============================================
Generates 4 full-length MP4 videos (requires ffmpeg):

  1. training_process.mp4
     The ENTIRE training run visualised - every episode plays out on the
     graph in real time, with live reward/success curves updating alongside.
     You watch agents go from random chaos -> learned coordination.

  2. trained_agents_demo.mp4
     Long showcase of the FULLY TRAINED policy running many episodes back-
     to-back on the graph. Shows smooth, intelligent agent behaviour.

  3. police_catches_thief.mp4
     Extended pursuit-only video: police hunting criminals across the graph,
     with chase trails, distance gauge, and catch highlights.

  4. baseline_comparison.mp4
     Animated line-chart race showing cumulative performance of MARL vs
     Greedy vs Random vs Static over all evaluation episodes.

Install ffmpeg (Windows):  winget install ffmpeg
Then restart your terminal before running.

Usage:
    from VideoGenerator import VideoGenerator
    vg = VideoGenerator(env, learner, comparator)
    vg.generate_all_videos()
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

RESPONDER_COLOR = "#74b9ff"
CRIMINAL_COLOR  = "#ff6b81"
NODE_COLOR      = "#636e72"
EDGE_COLOR      = "#2d3436"
BG_COLOR        = "#0d0d1a"
PANEL_COLOR     = "#12122a"
METHOD_COLORS   = {
    "marl":   "#2ed573",
    "greedy": "#74b9ff",
    "random": "#ff4757",
    "static": "#636e72",
}


class VideoGenerator:

    def __init__(self, env, learner, comparator, fps=15, dpi=100):
        self.env        = env
        self.learner    = learner
        self.comparator = comparator
        self.fps        = fps
        self.dpi        = dpi

        self.pos = {node: env.node_positions[node] for node in env.g_env.nodes()}

        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError(
                "\n\nffmpeg not found!\n"
                "Install with:  winget install ffmpeg\n"
                "Then CLOSE and REOPEN your terminal and run again.\n"
            )
        print("ffmpeg found, saving as mp4")

    def _save_mp4(self, ani, name):
        out = os.path.join(VIDEO_DIR, f"{name}.mp4")
        writer = animation.FFMpegWriter(
            fps=self.fps,
            metadata={"title": name},
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-preset", "fast"]
        )
        ani.save(out, writer=writer, dpi=self.dpi)
        plt.close("all")
        size_mb = os.path.getsize(out) / 1e6
        print(f"  Saved -> {out}  ({size_mb:.1f} MB)")
        return out

    def _draw_static_graph(self, ax):
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ax.set_aspect("equal")
        ax.axis("off")
        for u, v in self.env.g_env.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            ax.plot([x0, x1], [y0, y1], color=EDGE_COLOR, alpha=0.6, lw=1.2, zorder=1)
        xs = [self.pos[n][0] for n in self.env.g_env.nodes()]
        ys = [self.pos[n][1] for n in self.env.g_env.nodes()]
        ax.scatter(xs, ys, s=120, c=NODE_COLOR, edgecolors="#b2bec3",
                   linewidths=1, zorder=2, alpha=0.85)

    def _legend(self, ax):
        ax.legend(handles=[
            Line2D([0],[0], marker="o", color="w", markerfacecolor=RESPONDER_COLOR,
                   markersize=9, label="Police"),
            Line2D([0],[0], marker="X", color="w", markerfacecolor=CRIMINAL_COLOR,
                   markersize=9, label="Thief"),
        ], loc="upper right", facecolor=PANEL_COLOR, labelcolor="white",
           fontsize=8, framealpha=0.95)

    def _style_ax(self, ax, ylabel, xlabel="Episode", ylim=None):
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors="#b2bec3", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#2d3436")
        ax.set_ylabel(ylabel, color="#b2bec3", fontsize=8)
        ax.set_xlabel(xlabel, color="#b2bec3", fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.12, color="white")

    # =========================================================================
    # PUBLIC
    # =========================================================================
    def generate_all_videos(self):
        print("-" * 50)
        print("generating 4 videos...")
        print("-" * 50)
        self.video_training_process()
        self.video_trained_agents_demo()
        self.video_police_catches_thief()
        self.video_baseline_comparison()
        print("-" * 50)
        print("all videos saved to videos/")
        print()

    # =========================================================================
    # VIDEO 1 - FULL TRAINING PROCESS
    # =========================================================================
    def video_training_process(self):
        print("\n[1/4] Building training_process.mp4 ...")
        print("      Replaying all training episodes - this takes a few minutes.")

        n_episodes = len(self.learner.episode_rewards)

        sample_idx = np.linspace(0, n_episodes - 1,
                                 min(500, n_episodes), dtype=int)

        print(f"      Sampling {len(sample_idx)} episodes ...")
        snapshots = []

        for ep_i in sample_idx:
            state = self.env.reset()
            # FIX: pass env.agents (active agents) not active_emergencies
            sk = self.learner.get_state_key(state, self.env.agents)

            ep_epsilon = max(self.learner.epsilon_min,
                             self.learner.epsilon *
                             (self.learner.epsilon_decay ** ep_i))
            orig_eps = self.learner.epsilon
            self.learner.epsilon = ep_epsilon

            step_frames = []
            for _ in range(self.env.max_steps):
                step_frames.append({
                    "state":    dict(self.env.state),
                    "resolved": self.env.events_resolved,
                    "failed":   self.env.events_failed,
                    "catches":  self.env.catches,
                    "step":     self.env.current_time,
                    "ep":       ep_i,
                })
                acts = {a: self.learner.choose_action(a, sk, training=True)
                        for a in self.env.agents}
                _, _, terms, _, _ = self.env.step(acts)
                # FIX: pass env.agents after step
                sk = self.learner.get_state_key(self.env.state, self.env.agents)
                if any(terms.values()):
                    break

            self.learner.epsilon = orig_eps
            step_frames = step_frames[::5] or step_frames[:1]
            snapshots.extend(step_frames)

        print(f"      Total animation frames: {len(snapshots)}")

        window   = 50
        rewards  = pd.Series(self.learner.episode_rewards).rolling(
                       window, min_periods=1).mean().values
        success  = pd.Series(self.learner.success_rates).rolling(
                       window, min_periods=1).mean().values
        epsilons = [max(self.learner.epsilon_min,
                        self.learner.epsilon *
                        (self.learner.epsilon_decay ** i))
                    for i in range(n_episodes)]

        fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                left=0.03, right=0.97,
                                top=0.93,  bottom=0.06,
                                wspace=0.28, hspace=0.55)
        ax_graph  = fig.add_subplot(gs[:, 0])
        ax_reward = fig.add_subplot(gs[0, 1])
        ax_succ   = fig.add_subplot(gs[1, 1])
        ax_eps    = fig.add_subplot(gs[2, 1])

        fig.suptitle("MARL Police-Thief - Full Training Process",
                     color="white", fontsize=14, weight="bold")

        self._draw_static_graph(ax_graph)
        self._legend(ax_graph)

        resp_sc   = ax_graph.scatter([], [], s=280, c=RESPONDER_COLOR,
                                     marker="o", zorder=5,
                                     edgecolors="white", linewidths=1.2)
        crim_sc   = ax_graph.scatter([], [], s=280, c=CRIMINAL_COLOR,
                                     marker="X", zorder=5,
                                     edgecolors="white", linewidths=1.2)
        ep_txt    = ax_graph.text(0.5, 1.015, "", transform=ax_graph.transAxes,
                                  ha="center", color="white",
                                  fontsize=10, weight="bold")
        score_txt = ax_graph.text(0.02, 0.02, "", transform=ax_graph.transAxes,
                                  fontsize=8, color="#dfe6e9", va="bottom",
                                  bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor=PANEL_COLOR, alpha=0.9))

        self._style_ax(ax_reward, "Avg Reward (50-ep)")
        self._style_ax(ax_succ,   "Catch Rate",     ylim=(0, 1))
        self._style_ax(ax_eps,    "Exploration e",  ylim=(0, self.learner.epsilon))
        for ax in (ax_reward, ax_succ, ax_eps):
            ax.set_xlim(0, n_episodes)

        r_line, = ax_reward.plot([], [], color=METHOD_COLORS["marl"],   lw=1.8)
        s_line, = ax_succ.plot  ([], [], color=METHOD_COLORS["greedy"], lw=1.8)
        e_line, = ax_eps.plot   ([], [], color="#fdcb6e",               lw=1.8)
        r_fill  = ax_reward.fill_between([], [], alpha=0)

        def _update(fi):
            nonlocal r_fill
            f  = snapshots[fi]
            ep = int(f["ep"])

            rx = [self.pos[f["state"][a]][0] for a in self.env.possible_responders
                  if f["state"].get(a) is not None]
            ry = [self.pos[f["state"][a]][1] for a in self.env.possible_responders
                  if f["state"].get(a) is not None]
            resp_sc.set_offsets(np.c_[rx, ry] if rx else np.empty((0, 2)))

            cx = [self.pos[f["state"][a]][0] for a in self.env.possible_criminals
                  if f["state"].get(a) is not None]
            cy = [self.pos[f["state"][a]][1] for a in self.env.possible_criminals
                  if f["state"].get(a) is not None]
            crim_sc.set_offsets(np.c_[cx, cy] if cx else np.empty((0, 2)))

            ep_txt.set_text(f"Episode {ep+1}/{n_episodes}  |  Step {f['step']}")
            score_txt.set_text(
                f"Catches: {f['catches']}   Resolved: {f['resolved']}   Failed: {f['failed']}")

            end = min(ep + 1, n_episodes)
            r_line.set_data(range(end), rewards[:end])
            s_line.set_data(range(end), success[:end])
            e_line.set_data(range(end), epsilons[:end])

            if end > 1:
                ymin = min(rewards[:end])
                ymax = max(rewards[:end])
                pad  = (ymax - ymin) * 0.1 or 1
                ax_reward.set_ylim(ymin - pad, ymax + pad)

            try:
                r_fill.remove()
            except Exception:
                pass
            r_fill = ax_reward.fill_between(
                range(end), rewards[:end], alpha=0.15,
                color=METHOD_COLORS["marl"])

            return (resp_sc, crim_sc, ep_txt, score_txt,
                    r_line, s_line, e_line)

        ani = animation.FuncAnimation(
            fig, _update, frames=len(snapshots),
            interval=1000 // self.fps, blit=False)
        self._save_mp4(ani, "training_process")

    # =========================================================================
    # VIDEO 2 - LONG TRAINED AGENT DEMO
    # =========================================================================
    def video_trained_agents_demo(self, num_episodes=20):
        print(f"\n[2/4] Building trained_agents_demo.mp4 ({num_episodes} episodes) ...")

        all_frames          = []
        cumulative_resolved = 0
        cumulative_failed   = 0
        cumulative_reward   = 0
        ep_rewards          = []

        for ep in range(num_episodes):
            print(f"      Episode {ep+1}/{num_episodes} ...", end="\r")
            state = self.env.reset()
            # FIX: pass env.agents not active_emergencies
            sk    = self.learner.get_state_key(state, self.env.agents)
            ep_reward = 0

            for _ in range(self.env.max_steps):
                all_frames.append({
                    "state":      dict(self.env.state),
                    "resolved":   self.env.events_resolved,
                    "failed":     self.env.events_failed,
                    "catches":    self.env.catches,
                    "step":       self.env.current_time,
                    "ep":         ep,
                    "cum_res":    cumulative_resolved + self.env.events_resolved,
                    "cum_fail":   cumulative_failed   + self.env.events_failed,
                    "cum_rew":    cumulative_reward   + ep_reward,
                    "ep_rewards": list(ep_rewards),
                })
                acts = {a: self.learner.choose_action(a, sk, training=False)
                        for a in self.env.agents}
                _, rews, terms, _, _ = self.env.step(acts)
                ep_reward += sum(rews.values())
                # FIX: pass env.agents not active_emergencies
                sk = self.learner.get_state_key(self.env.state, self.env.agents)
                if any(terms.values()):
                    break

            cumulative_resolved += self.env.events_resolved
            cumulative_failed   += self.env.events_failed
            cumulative_reward   += ep_reward
            ep_rewards.append(ep_reward)

        print(f"\n      Total frames: {len(all_frames)}")

        fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                left=0.03, right=0.97,
                                top=0.93,  bottom=0.06,
                                wspace=0.28, hspace=0.45)
        ax_graph  = fig.add_subplot(gs[:, 0])
        ax_ep_rew = fig.add_subplot(gs[0, 1])
        ax_stats  = fig.add_subplot(gs[1, 1])

        fig.suptitle("MARL Police-Thief - Trained Policy Demonstration",
                     color="white", fontsize=14, weight="bold")

        self._draw_static_graph(ax_graph)
        self._legend(ax_graph)

        resp_sc   = ax_graph.scatter([], [], s=300, c=RESPONDER_COLOR,
                                     marker="o", zorder=5,
                                     edgecolors="white", linewidths=1.3)
        crim_sc   = ax_graph.scatter([], [], s=300, c=CRIMINAL_COLOR,
                                     marker="X", zorder=5,
                                     edgecolors="white", linewidths=1.3)
        ep_txt    = ax_graph.text(0.5, 1.015, "", transform=ax_graph.transAxes,
                                  ha="center", color="white",
                                  fontsize=10, weight="bold")
        score_txt = ax_graph.text(0.02, 0.02, "", transform=ax_graph.transAxes,
                                  fontsize=8, color="#dfe6e9", va="bottom",
                                  bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor=PANEL_COLOR, alpha=0.9))

        self._style_ax(ax_ep_rew, "Episode Reward")
        ax_ep_rew.set_xlim(-0.5, num_episodes - 0.5)
        rew_bars     = ax_ep_rew.bar(range(num_episodes), [0] * num_episodes,
                                     color=METHOD_COLORS["marl"], alpha=0.8)
        rew_mean_ln, = ax_ep_rew.plot([], [], color="#fdcb6e", lw=2,
                                      linestyle="--", label="Running mean")
        ax_ep_rew.legend(facecolor=PANEL_COLOR, labelcolor="white", fontsize=7)

        ax_stats.set_facecolor(PANEL_COLOR)
        ax_stats.axis("off")
        stats_txt = ax_stats.text(
            0.5, 0.5, "", transform=ax_stats.transAxes,
            ha="center", va="center", fontsize=12, color="white",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#1e1e3a", alpha=0.95))

        def _update(fi):
            f  = all_frames[fi]
            ep = f["ep"]

            rx = [self.pos[f["state"][a]][0] for a in self.env.possible_responders
                  if f["state"].get(a) is not None]
            ry = [self.pos[f["state"][a]][1] for a in self.env.possible_responders
                  if f["state"].get(a) is not None]
            resp_sc.set_offsets(np.c_[rx, ry] if rx else np.empty((0, 2)))

            cx = [self.pos[f["state"][a]][0] for a in self.env.possible_criminals
                  if f["state"].get(a) is not None]
            cy = [self.pos[f["state"][a]][1] for a in self.env.possible_criminals
                  if f["state"].get(a) is not None]
            crim_sc.set_offsets(np.c_[cx, cy] if cx else np.empty((0, 2)))

            ep_txt.set_text(f"Episode {ep+1}/{num_episodes}  |  Step {f['step']}")
            score_txt.set_text(
                f"Catches: {f['catches']}  Resolved: {f['resolved']}  Failed: {f['failed']}")

            for i, (bar, r) in enumerate(zip(rew_bars, f["ep_rewards"])):
                bar.set_height(r)
            if f["ep_rewards"]:
                all_r = f["ep_rewards"]
                ax_ep_rew.set_ylim(min(0, min(all_r)) * 1.1,
                                   max(all_r) * 1.15 or 1)
                means = np.cumsum(all_r) / np.arange(1, len(all_r) + 1)
                rew_mean_ln.set_data(range(len(means)), means)

            total = f["cum_res"] + f["cum_fail"]
            sr    = f["cum_res"] / total if total else 0
            stats_txt.set_text(
                f"CUMULATIVE STATS\n\n"
                f"Episodes     : {ep+1:>4}\n"
                f"Catches      : {f['catches']:>4}\n"
                f"Resolved     : {f['cum_res']:>4}\n"
                f"Failed       : {f['cum_fail']:>4}\n"
                f"Success Rate : {sr:>6.1%}\n"
                f"Total Reward : {f['cum_rew']:>8.1f}")

            return (resp_sc, crim_sc, ep_txt, score_txt,
                    rew_mean_ln, stats_txt)

        ani = animation.FuncAnimation(
            fig, _update, frames=len(all_frames),
            interval=1000 // self.fps, blit=False)
        self._save_mp4(ani, "trained_agents_demo")

    # =========================================================================
    # VIDEO 3 - POLICE CHASING THIEF
    # =========================================================================
    def video_police_catches_thief(self, num_episodes=15):
        print(f"\n[3/4] Building police_catches_thief.mp4 ({num_episodes} episodes) ...")

        all_frames   = []
        catch_per_ep = [0] * num_episodes

        for ep in range(num_episodes):
            print(f"      Episode {ep+1}/{num_episodes} ...", end="\r")
            state = self.env.reset()
            # FIX: pass env.agents not active_emergencies
            sk    = self.learner.get_state_key(state, self.env.agents)

            for _ in range(self.env.max_steps):
                police_pos = {r: self.env.state[r]
                              for r in self.env.possible_responders}
                thief_pos  = {c: self.env.state[c]
                              for c in self.env.possible_criminals}

                min_d     = 999
                best_path = None
                for rnode in police_pos.values():
                    for cnode in thief_pos.values():
                        if rnode and cnode:
                            try:
                                p = nx.shortest_path(self.env.g_env, rnode, cnode)
                                if len(p) - 1 < min_d:
                                    min_d     = len(p) - 1
                                    best_path = p
                            except Exception:
                                pass

                caught = [c for c in self.env.possible_criminals
                          if any(self.env.state[r] == self.env.state[c]
                                 for r in self.env.possible_responders)]
                if caught:
                    catch_per_ep[ep] += len(caught)

                all_frames.append({
                    "police":       police_pos,
                    "thieves":      thief_pos,
                    "min_dist":     min_d if min_d < 999 else 0,
                    "best_path":    best_path,
                    "caught":       caught,
                    "step":         self.env.current_time,
                    "ep":           ep,
                    "catch_per_ep": list(catch_per_ep),
                })

                acts = {a: self.learner.choose_action(a, sk, training=False)
                        for a in self.env.agents}
                _, _, terms, _, _ = self.env.step(acts)
                # FIX: pass env.agents not active_emergencies
                sk = self.learner.get_state_key(self.env.state, self.env.agents)
                if any(terms.values()):
                    break

        print(f"\n      Total frames: {len(all_frames)}")

        fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                left=0.03, right=0.97,
                                top=0.93,  bottom=0.06,
                                wspace=0.28, hspace=0.45)
        ax_graph = fig.add_subplot(gs[:, 0])
        ax_dist  = fig.add_subplot(gs[0, 1])
        ax_catch = fig.add_subplot(gs[1, 1])

        fig.suptitle("MARL Pursuit - Police vs Thief",
                     color="white", fontsize=14, weight="bold")

        self._draw_static_graph(ax_graph)

        path_lines = [ax_graph.plot([], [], color="#fdcb6e", lw=2.5,
                                    alpha=0.6, linestyle="--", zorder=3)[0]
                      for _ in self.env.possible_criminals]
        police_sc  = ax_graph.scatter([], [], s=350, c=RESPONDER_COLOR,
                                      marker="o", zorder=6,
                                      edgecolors="white", lw=1.5, label="Police")
        thief_sc   = ax_graph.scatter([], [], s=350, c=CRIMINAL_COLOR,
                                      marker="X", zorder=6,
                                      edgecolors="white", lw=1.5, label="Thief")
        catch_sc   = ax_graph.scatter([], [], s=1200, c="#f1c40f",
                                      marker="*", zorder=7, alpha=0)
        title_txt  = ax_graph.text(0.5, 1.015, "", transform=ax_graph.transAxes,
                                   ha="center", color="white",
                                   fontsize=10, weight="bold")
        ax_graph.legend(facecolor=PANEL_COLOR, labelcolor="white",
                        fontsize=9, loc="lower right")

        self._style_ax(ax_dist, "Min Graph Distance", xlabel="Frame", ylim=(0, 15))
        ax_dist.set_xlim(0, len(all_frames))
        ax_dist.axhline(0, color="#ff4757", lw=1.5, linestyle="--",
                        alpha=0.6, label="Caught (dist=0)")
        ax_dist.legend(facecolor=PANEL_COLOR, labelcolor="white", fontsize=7)
        dist_line, = ax_dist.plot([], [], color="#a29bfe", lw=1.8)
        dist_fill  = ax_dist.fill_between([], [], alpha=0)

        self._style_ax(ax_catch, "Catches per Episode")
        ax_catch.set_xlim(-0.5, num_episodes - 0.5)
        ax_catch.set_ylim(0, max(len(self.env.possible_criminals) * 3, 1))
        catch_bars = ax_catch.bar(range(num_episodes), [0] * num_episodes,
                                  color=CRIMINAL_COLOR, alpha=0.8)

        dist_history = []

        def _update(fi):
            nonlocal dist_fill
            f  = all_frames[fi]
            ep = f["ep"]

            px = [self.pos[f["police"][r]][0] for r in self.env.possible_responders
                  if f["police"].get(r) is not None]
            py = [self.pos[f["police"][r]][1] for r in self.env.possible_responders
                  if f["police"].get(r) is not None]
            police_sc.set_offsets(np.c_[px, py] if px else np.empty((0, 2)))

            tx = [self.pos[f["thieves"][c]][0] for c in self.env.possible_criminals
                  if f["thieves"].get(c) is not None]
            ty = [self.pos[f["thieves"][c]][1] for c in self.env.possible_criminals
                  if f["thieves"].get(c) is not None]
            thief_sc.set_offsets(np.c_[tx, ty] if tx else np.empty((0, 2)))

            if f["best_path"] and len(path_lines) > 0:
                lx = [self.pos[n][0] for n in f["best_path"]]
                ly = [self.pos[n][1] for n in f["best_path"]]
                path_lines[0].set_data(lx, ly)
            else:
                path_lines[0].set_data([], [])

            if f["caught"]:
                cx2 = [self.pos[f["thieves"][c]][0] for c in f["caught"]
                       if f["thieves"].get(c) is not None]
                cy2 = [self.pos[f["thieves"][c]][1] for c in f["caught"]
                       if f["thieves"].get(c) is not None]
                catch_sc.set_offsets(
                    np.c_[cx2, cy2] if cx2 else np.empty((0, 2)))
                catch_sc.set_alpha(0.9)
            else:
                catch_sc.set_offsets(np.empty((0, 2)))
                catch_sc.set_alpha(0)

            title_txt.set_text(
                f"Episode {ep+1}/{num_episodes}  |  Step {f['step']}"
                + ("   CAUGHT!" if f["caught"] else ""))

            dist_history.append(f["min_dist"])
            dist_line.set_data(range(len(dist_history)), dist_history)
            try:
                dist_fill.remove()
            except Exception:
                pass
            dist_fill = ax_dist.fill_between(
                range(len(dist_history)), dist_history,
                alpha=0.2, color="#a29bfe")

            for i, bar in enumerate(catch_bars):
                bar.set_height(f["catch_per_ep"][i])

            return ([police_sc, thief_sc, catch_sc, title_txt, dist_line]
                    + path_lines + list(catch_bars))

        ani = animation.FuncAnimation(
            fig, _update, frames=len(all_frames),
            interval=1000 // self.fps, blit=False)
        self._save_mp4(ani, "police_catches_thief")

    # =========================================================================
    # VIDEO 4 - BASELINE COMPARISON
    # =========================================================================
    def video_baseline_comparison(self):
        print("\n[4/4] Building baseline_comparison.mp4 ...")

        results = self.comparator.results
        methods = ["marl", "greedy", "random", "static"]
        n_ep    = min(len(results[m]["rewards"]) for m in methods)

        def cum_mean(lst):
            a = np.array(lst[:n_ep], dtype=float)
            return np.cumsum(a) / (np.arange(len(a)) + 1)

        def cum_mean_nonzero(lst):
            out, s, c = [], 0, 0
            for v in lst[:n_ep]:
                if v > 0:
                    s += v; c += 1
                out.append(s / c if c else 0)
            return np.array(out)

        reward_cm   = {m: cum_mean(results[m]["rewards"])               for m in methods}
        response_cm = {m: cum_mean_nonzero(results[m]["response_times"]) for m in methods}
        success_cm  = {m: cum_mean(results[m]["success_rates"])          for m in methods}

        fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(3, 1, figure=fig,
                                left=0.08, right=0.97,
                                top=0.91,  bottom=0.08,
                                hspace=0.55)
        axes = [fig.add_subplot(gs[i]) for i in range(3)]

        fig.suptitle("MARL vs Baseline Methods - Cumulative Performance Race",
                     color="white", fontsize=14, weight="bold")

        metric_cfg = [
            (axes[0], reward_cm,   "Avg Reward",        True),
            (axes[1], response_cm, "Avg Response Time", False),
            (axes[2], success_cm,  "Success Rate",      True),
        ]

        all_lines  = []
        best_texts = []

        for ax, data, ylabel, higher_better in metric_cfg:
            self._style_ax(ax, ylabel)
            ax.set_xlim(0, n_ep)

            all_vals = np.concatenate([data[m] for m in methods])
            ypad     = (all_vals.max() - all_vals.min()) * 0.1 or 0.1
            ax.set_ylim(all_vals.min() - ypad, all_vals.max() + ypad)

            lines = {}
            for m in methods:
                ln, = ax.plot([], [], color=METHOD_COLORS[m], lw=2.2,
                              label=m.upper())
                lines[m] = ln
            ax.legend(facecolor=PANEL_COLOR, labelcolor="white",
                      fontsize=8, loc="upper right" if higher_better else "upper left")
            all_lines.append((lines, data))

            bt = ax.text(0.01, 0.96, "", transform=ax.transAxes,
                         va="top", fontsize=8, color="#f1c40f",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor=PANEL_COLOR, alpha=0.85))
            best_texts.append((bt, data, higher_better))

        ep_txt = fig.text(0.5, 0.003, "", ha="center",
                          color="#b2bec3", fontsize=10)

        def _update(fi):
            end = fi + 1
            ep_txt.set_text(f"Episode: {end} / {n_ep}")
            artists = [ep_txt]
            for (lines, data), (bt, bdata, hb) in zip(all_lines, best_texts):
                for m in methods:
                    lines[m].set_data(range(end), data[m][:end])
                    artists.append(lines[m])
                cur  = {m: data[m][fi] for m in methods}
                best = (max if hb else min)(cur, key=cur.get)
                bt.set_text(f"Leading: {best.upper()}  ({cur[best]:.2f})")
                artists.append(bt)
            return artists

        ani = animation.FuncAnimation(
            fig, _update, frames=n_ep,
            interval=1000 // self.fps, blit=True)
        self._save_mp4(ani, "baseline_comparison")


if __name__ == "__main__":
    print("VideoGenerator ready. Import and call generate_all_videos().")
