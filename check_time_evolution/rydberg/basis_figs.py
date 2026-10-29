"""Figures of the local-basis optimisation for R1 at Omega/V_nn = 0.1 (basis_opt.py).

usage: python basis_figs.py   -> figs/basis_proxies.pdf, figs/basis_angles.pdf
"""
import glob, os, sys

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quench2d"))
from plotstyle import STYLE, COLORS, simple_ticks, panel, plt

B, RBM = "data/basis", "data/rbm/R1_sq4x3_+x_+0.1000_rbm1"
FLOOR = -30.0
LAM_SHOW = "0.01"  # spline whose angles are drawn

# --- data --------------------------------------------------------------------------
fixed = np.load(f"{B}/traj_fixed.npz")
splines = {lam: np.load(f"{B}/traj_spline_lam{lam}.npz") for lam in ("0.01", "1")}
snaps = {T: np.load(f"{B}/snap_t{T}.npz") for T in ("1.65", "4.95", "30")}
bases = [("z", r"$\sigma^z$ basis", COLORS[0]), ("traj_fixed", r"fixed $U$", COLORS[1]),
         ("traj_spline_lam0.01", r"$U(t)$, $\lambda=0.01$", COLORS[2]),
         ("traj_spline_lam1", r"$U(t)$, $\lambda=1$", COLORS[3])]
proxies = {"z": {k: fixed[f"val_z_{k}"] for k in ("t", "mu", "mu_ph")},
           "traj_fixed": {k: fixed[f"val_opt_{k}"] for k in ("t", "mu", "mu_ph")}}
for lam, d in splines.items():
    proxies[f"traj_spline_lam{lam}"] = {k: d[f"val_{k}"] for k in ("t", "mu", "mu_ph")}


def tdvp(name):
    return np.load(f"{RBM}.npz") if name == "z" else np.load(f"{RBM}_{name}.npz")


def sup(name):
    """Supervised chains of all seeds: t, (seeds, t) infidelities."""
    runs = [np.load(f) for f in sorted(glob.glob(f"{B}/sup_{name}_s*_a1.npz"))]
    return runs[0]["t"], np.array([r["inf"] for r in runs])


# %% mu, mu_ph and the RBM infidelity per basis
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, layout="constrained")
    for name, label, color in bases:
        p = proxies[name]
        axs[0, 0].plot(p["t"], np.maximum(p["mu"], FLOOR), color=color, lw=1, label=label)
        axs[0, 1].plot(p["t"], np.maximum(p["mu_ph"], FLOOR), color=color, lw=1, label=label)
        r = tdvp(name)
        axs[1, 0].plot(r["t"], np.log10(r["inf"]), color=color, lw=1, label=label)
        t, I = sup(name)
        axs[1, 1].plot(t[1:], np.log10(np.median(I[:, 1:], axis=0)), color=color, lw=1, label=label)
        axs[1, 1].fill_between(t[1:], np.log10(I[:, 1:].min(0)), np.log10(I[:, 1:].max(0)), color=color, alpha=0.2, lw=0)
    for T, d in snaps.items():
        for ax, key in ((axs[0, 0], "val_mu"), (axs[0, 1], "val_mu_ph")):
            ax.plot(float(T), d[key][0], "*", color=COLORS[5], mec="black", mew=0.4, ms=9, zorder=5,
                    label="per-time optimal $U$" if T == "1.65" else None)
    axs[0, 0].set(ylabel=r"$\mu$", ylim=(-16, -3))
    axs[0, 1].set(ylabel=r"$\mu_{\rm ph}$", ylim=(-16, -3))
    for ax in axs[1]:
        ax.set(ylim=(-9, 0))
    for ax in axs[0]:
        ax.axhline(-10, ls=":", color="gray", lw=0.8, label=r"onset threshold $-10$")
    axs[1, 0].set(ylabel=r"$\log_{10} I$ (TDVP)", xlabel=r"$t\,V_{nn}$")
    axs[1, 1].set(ylabel=r"$\log_{10} I$ (supervised fit)", xlabel=r"$t\,V_{nn}$")
    for ax in axs[1]:
        ax.axhline(-3, ls="--", color="gray", lw=0.8, label=r"$I=10^{-3}$")
    axs[0, 0].legend(loc="upper left")  # empty before the proxies rise
    axs[1, 0].legend(loc="lower right")
    for i, ax in enumerate(axs.flat):
        ax.set_xscale("log")  # the bases differ mainly at t <~ 3
        ax.set_xlim(0.15, 34)  # room for the t = 30 stars
        simple_ticks(ax, x=[0.2, 0.5, 1, 2, 5, 10, 30])
        ax.set_title(panel(i), loc="left")
    fig.savefig("figs/basis_proxies.pdf"); fig.savefig("figs/basis_proxies.png", dpi=110)

# %% angles of the optimal frames and the uniform-basis landscape
from scipy.interpolate import CubicSpline
from basis_opt import KNOTS, LY, N
row_colors = COLORS[:LY]


def canonical(th, ph):
    """Equivalent axis with 0 <= theta <= pi/2, -pi < phi <= pi: n ~ -n, and (-theta, phi) ~ (theta, phi + pi)."""
    th = (np.asarray(th) + np.pi) % (2 * np.pi) - np.pi
    th = np.where(np.abs(th) > np.pi / 2, th - np.sign(th) * np.pi, th)
    ph = np.where(th < 0, np.asarray(ph) + np.pi, ph)
    return np.abs(th), (ph + np.pi) % (2 * np.pi) - np.pi


def broken(y):
    """NaN at jumps of more than pi, so wrapped angles are not joined by vertical lines."""
    y = np.array(y, float)
    y[1:][np.abs(np.diff(y)) > np.pi] = np.nan
    return y


with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
    tt = np.linspace(0, 30, 301)
    v = CubicSpline(KNOTS, splines[LAM_SHOW]["xk"].reshape(len(KNOTS), -1))(tt)
    th_t, ph_t = canonical(v[:, :N], v[:, N:])
    th_f, ph_f = canonical(fixed["x"][0][:N], fixed["x"][0][N:])
    th_s = {T: canonical(d["x"][0][:N], d["x"][0][N:]) for T, d in snaps.items()}
    for i in range(N):
        y = i % LY
        lab = rf"row $y={y}$" if i < LY else None
        axs[0, 0].plot(tt, th_t[:, i], color=row_colors[y], lw=0.8, label=lab)
        axs[0, 1].plot(tt, broken(ph_t[:, i]), color=row_colors[y], lw=0.8, label=lab)
        axs[0, 0].axhline(th_f[i], color=row_colors[y], ls="--", lw=0.6)
        axs[0, 1].axhline(ph_f[i], color=row_colors[y], ls="--", lw=0.6)
        for T in snaps:
            axs[0, 0].plot(float(T), th_s[T][0][i], "o", mfc="none", color=row_colors[y], ms=4)
            axs[0, 1].plot(float(T), th_s[T][1][i], "o", mfc="none", color=row_colors[y], ms=4)
    axs[0, 0].set(ylabel=r"$\theta_i$", xlabel=r"$t\,V_{nn}$")
    axs[0, 1].set(ylabel=r"$\phi_i$", xlabel=r"$t\,V_{nn}$", ylim=(-np.pi, 2 * np.pi))  # headroom for the legend
    axs[0, 1].plot([], [], "--", color="gray", lw=0.6, label=r"fixed $U$")
    axs[0, 1].plot([], [], "o", mfc="none", color="gray", ms=4, label="per-time optimal $U$")
    axs[0, 1].legend(loc="upper right")
    for ax, T in zip(axs[1], ("1.65", "30")):
        d = snaps[T]
        m = ax.pcolormesh(d["grid_ph"], d["grid_th"], np.clip(d["grid_J"], None, 0), cmap="viridis", shading="nearest")
        cb = fig.colorbar(m, ax=ax, label=rf"$\mu_{{\max}}(t={T})$, uniform $U$")
        simple_ticks(cb.ax)
        ax.plot(0, 0, "x", color="white", ms=6, label=r"$\sigma^z$ basis")
        ax.legend(loc="lower right")
        ax.set(xlabel=r"$\phi$", ylabel=r"$\theta$")
    for i, ax in enumerate(axs.flat):
        simple_ticks(ax)
        ax.set_title(panel(i), loc="left")
    fig.savefig("figs/basis_angles.pdf"); fig.savefig("figs/basis_angles.png", dpi=110)
