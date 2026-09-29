"""Diagnostic plots for the 12D effective-gain Crazyflie rollout."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_LABELS = ("p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "roll", "pitch", "yaw", "p", "q", "r")


def save_diagnostic_plots(checkpoint: str | Path) -> None:
    """Write hardware-quadrotor-style diagnostics beside a saved rollout."""
    checkpoint = Path(checkpoint)
    out = checkpoint.parent
    data = np.load(checkpoint, allow_pickle=True)
    xs = data["xs"]
    beta = data["beta_estimates"][:, 0]
    width = data["beta_widths"][:, 0]
    dt = float(data["controller_dt"])
    t = np.arange(len(xs)) * dt
    controls = data["controls"]
    goal = data["goal_position"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, beta, label=r"$\hat\beta$")
    ax.fill_between(t, beta - width, beta + width, alpha=.18, label="uncertainty interval")
    ax.axhline(float(data["true_beta"]), color="tab:red", linestyle="--", label=r"true $\beta$ (simulation)")
    ax.set(xlabel="time (s)", ylabel=r"effective gain $\beta$", title="Effective thrust-gain estimate")
    ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig(out / "crazyflie_beta_estimate.png", dpi=250); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, xs[:, i], label="executed")
        ax.axhline(goal[i], color="tab:red", linestyle="--", label="goal")
        ax.set_ylabel(f"{('x', 'y', 'z')[i]} (m)"); ax.grid(); ax.legend()
    axes[-1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(out / "crazyflie_xyz_goal_regulation.png", dpi=250); plt.close(fig)

    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], linewidth=2, label="executed")
    ax.scatter(*xs[0, :3], color="black", marker="o", label="start")
    ax.scatter(*goal, color="tab:red", marker="x", s=55, label="goal")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="12D Crazyflie goal-directed rollout")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "crazyflie_xyz_rollout.png", dpi=250); plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    labels = ("thrust (N)", "roll command (rad)", "pitch command (rad)", "yaw-rate command (rad/s)")
    tc = np.arange(len(controls)) * dt
    for i, ax in enumerate(axes):
        ax.step(tc, controls[:, i], where="post")
        ax.set_ylabel(labels[i]); ax.grid()
    axes[-1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(out / "crazyflie_controls.png", dpi=250); plt.close(fig)

    errors = data["one_step_errors"]
    te = np.arange(len(errors)) * dt
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(te, errors[:, i]); ax.axhline(0., color="black", linewidth=.6)
        ax.set_title(STATE_LABELS[i]); ax.grid()
    for ax in axes[-1]: ax.set_xlabel("time (s)")
    fig.suptitle("Analytical-model one-step error", y=1.01); fig.tight_layout(); fig.savefig(out / "crazyflie_one_step_errors.png", dpi=250); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].step(te, data["estimator_ranks"], where="post"); axes[0].set(ylabel="rank", title="Velocity-gated estimator diagnostics"); axes[0].grid()
    axes[1].plot(te, data["estimator_innovations"][:, 0]); axes[1].set(xlabel="time (s)", ylabel="innovation"); axes[1].grid()
    fig.tight_layout(); fig.savefig(out / "crazyflie_estimator_diagnostics.png", dpi=250); plt.close(fig)
