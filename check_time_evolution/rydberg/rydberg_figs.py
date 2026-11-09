"""Figures of the Rydberg screen (PLAN_rydberg_quench.md steps 2, 3, 7).

usage: python rydberg_figs.py screen   -> figs/screen/<ID>_<lat>_<init>.png  quick-look sheets
       python rydberg_figs.py map      -> figs/map.pdf  overview map (needs rydberg_triage.py)
"""
import glob, os, sys

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quench2d"))  # quench2d.py, analysis.py, plotstyle.py
from plotstyle import STYLE, COLORS, simple_ticks, panel, plt, model_label
from rydberg2d import lattice

S_RANDOM = lambda N: N / 2 * np.log(2) - 0.5  # average half-cut entropy of a random state (Page); MPS-hard = half of it

FLOOR = -30.0
MU_LIM = (FLOOR, -2.0)  # shared colour limits for mu and mu_ph in every sheet
HAM = {  # one-line Hamiltonian + fixed parameters, for titles
    "S3": r"kicked Ising $J=b=\pi/4$, longitudinal $h$", "S3b": r"kicked Ising $J=\pi/4$, $h=0$ (IBM Eagle circuit)",
    "R0": r"Rydberg vdW, $\Omega=\Delta=0$", "R6": r"Rydberg dressing $U_0/(1+(r/R_c)^6)$, $\Omega=0.05$",
    "R1": r"Rydberg vdW, $\Delta=0$", "G4": r"$-\sum r^{-6} ZZ - h_f\sum X$ from gs($h=4$)",
    "R4": r"Rydberg vdW, $\Omega=1$, $\Delta=0$", "G1": r"Rydberg $R_b/a=1.15$, $\Omega=1$, from gs($\Delta=-2$)",
    "R5": r"Rydberg vdW, $V_{nn}=12\Omega$", "C2": r"PXP-like, $V_{nn}=12\Omega$, $\Delta(t)=\Omega(0.85+0.98\cos\omega_m t)$",
    "D1": r"dipolar XY, field tilted by $\Theta$", "G3": r"dipolar XXZ from gs(XY ferro $-0.2\sum X$)",
    "D3": r"dipolar XXZ", "D4": r"AFM dipolar XY + staggered $\delta\sum\eta_i Z$",
    "I1": r"$\sum r^{-\kappa}ZZ + 0.1\sum X$", "Q1": r"QAOA: MIS phase ($\gamma=1$) + mixer $\beta$, per layer",
    "R3": r"Rydberg vdW, $\Omega=0.1$", "C1": r"facilitation $\Delta=V_{nn}+\delta$, $\Omega=0.1$",
    "S1": r"toric code $+\epsilon\sum Z$", "S2": r"2D cluster $-\sum X\prod Z + \epsilon\sum X$",
    "S4": r"kicked Ising $J=\pi/4+\epsilon$, $b=\pi/4$", "I5": r"$\sum_\triangle ZZZ + h_x\sum X$",
    "R9": r"dual-species vdW, $\Omega=0.1$", "R8": r"Rydberg vdW + random $\Delta_i\in[-2,2]$",
    "I2": r"dipolar Ising $\sum r^{-3}ZZ + h_x\sum X$", "D2": r"dipolar XY ferro $+h_x\sum X$",
    "D5": r"dipolar XY $+\lambda\sum r^{-6}ZZ$", "G2": r"Rydberg $\Delta=3$ from checkerboard gs",
    "R2": r"Rydberg vdW, $\Delta=0$", "D6": r"diluted dipolar XY", "R7": r"Rydberg vdW, position disorder, $\Omega=0.1$",
    "I3": r"$-\sum ZZ - 0.25\sum X - h_z\sum Z$", "I4": r"NN Ising $+h_x\sum X$", "S5": r"XY chain (free fermions)",
    "S6": r"$\sum XX + \Omega\sum Z$", "S7": r"XXZ chain", "C3": r"quantum East",
    "B1": r"toric code $-h_f\sum Z$ from gs($h=0.1$)", "B2": r"toric code $-h_f\sum(Z+X)/\sqrt{2}$ from gs($h=0.1$)",
    "B3": r"Xu-Moore $-\sum_\boxdot ZZZZ-h\sum X$", "B4": r"Baxter-Wu $-\sum_\triangle ZZZ-h\sum X$",
    "B5": r"Newman-Moore $-\sum_{\triangle_\uparrow}ZZZ-h\sum X$", "B6": r"$-\sum ZZ-K\sum_\boxdot ZZZZ-0.5\sum X$",
    "B7": r"$-\sum_\triangle ZZZ-\lambda\sum ZZ-0.5\sum X$",
}
INIT = {"+x": r"$|{+}x\rangle$", "th": r"$|\theta\rangle$", "Z2th": r"$|\mathbb{Z}_2,\theta\rangle$", "gs": "ground state",
        "g~": r"$|g\rangle$ tilted 0.1", "120": r"$120^\circ$", "seed": "seed"}


def unit_label(unit):
    return unit if "period" in unit else f"1/{unit}"  # Floquet families count time in drive steps


def sheet(f):
    """mu | mu_ph | S | chi-fraction over (t, g) for one (family, lattice, init)."""
    d = np.load(f)
    key = os.path.basename(f)[:-4]
    ID, lat, init = key.split("_")[-3:]
    t, g = d["t"], d["g"]
    order = np.argsort(g)
    t_ends = np.nanmax(d["t_g"], axis=1)
    scaled = np.ptp(t_ends) > 1e-9  # t_max depends on g: show each row against t / t_max(g)
    quantities = [("mu", r"$\mu$", MU_LIM), ("mu_ph", r"$\mu_{\rm ph}$", MU_LIM),
                  ("S", r"$S_{N/2}$", (0, None)), ("chi_frac", r"$\chi_{\epsilon}/\chi_{\rm cap}$", (0, 1))]
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True, layout="constrained")
        for i, (ax, (name, label, lim)) in enumerate(zip(axs, quantities)):
            z = d[name][order].astype(float)
            if name.startswith("mu"):
                z = np.where(np.isnan(z), np.nan, np.maximum(z, FLOOR))  # -inf (exactly few-body) -> floor
            for row, k in enumerate(order):  # one strip per g: t_max can depend on g
                tk = d["t_g"][k][np.isfinite(d["t_g"][k])]
                tk = tk / tk[-1] if scaled else tk
                edges = np.concatenate([[tk[0]], (tk[1:] + tk[:-1]) / 2, [tk[-1]]])
                mesh = ax.pcolormesh(edges, [row - 0.5, row + 0.5], z[row:row + 1, :len(tk)], cmap="viridis",
                                     vmin=lim[0], vmax=lim[1])
                if name.startswith("mu") and (z[row, :len(tk)] > -10).any():  # t* = first crossing of -10
                    ax.plot(tk[np.argmax(z[row, :len(tk)] > -10)], row, "|", color="white", ms=6, mew=1.2)
            cbar = fig.colorbar(mesh, ax=ax, label=label)
            simple_ticks(cbar.ax)
            ax.set(xlabel=r"$t/t_{\max}(g)$" if scaled else rf"time $t$ [{unit_label(str(d['unit'][0]))}]")
            ax.set_title(panel(i), loc="left")
            simple_ticks(ax)
        axs[0].set_yticks(np.arange(len(g)), [f"${v:.3g}$" for v in g[order]])  # categorical: grid points
        axs[0].set_ylabel(str(d["glabel"]).replace(r"\alpha", r"\kappa"))
        for gm, lab in zip(d["marks_g"], d["marks_l"]):
            k = int(np.argmin(np.abs(g[order] - float(gm))))
            axs[0].annotate(r"exp.", (t[0], k), xytext=(-60, 0), textcoords="offset points", fontsize=6, color="red",
                            va="center", arrowprops=dict(arrowstyle="->", color="red", lw=0.6), annotation_clip=False)
        fig.suptitle(rf"{key.replace('_', ' ')}: {HAM.get(ID, ID)}; init {INIT.get(init, init)}; $N={int(d['N'][0])}$, $\eta={np.median(d['eta']):.2f}$"
                     r" (white bar: first $\mu=-10$ crossing)", fontsize=9)
        fig.savefig(f"figs/screen/{key}.png", dpi=90)
        plt.close(fig)


def overview_map():
    """Every screened point in the (max S, mu_ph) plane, coloured by eta. The proxies are maximised over t <= t_S (the time
    the state becomes MPS-hard), or over the whole run if S never gets there: at late times every non-classical run saturates."""
    df = pd.read_csv("data/triage.csv")
    df["S_max"] = df.S_rel_max * S_RANDOM(df.N)  # back to nats
    markers = {1: "o", 2: "s", 3: "^"}
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, layout="constrained")
        for ax, (col, xl) in zip(axs, [("muph_at_S", r"$\max_{t\leq t_S}\mu_{\rm ph}$"), ("mu_at_S", r"$\max_{t\leq t_S}\mu$")]):
            for tier, sub in df.groupby("tier"):
                sc = ax.scatter(sub[col], sub["S_max"], c=sub["eta"], cmap="viridis", vmin=0, vmax=1, s=10,
                                marker=markers[tier], edgecolors="none", label=f"Tier {tier}")
            ax.axvline(-10, ls="--", color="gray", lw=0.8, label=r"threshold $-10$")
            for N, ls in ((12, ":"), (16, "-.")):
                ax.axhline(S_RANDOM(N) / 2, ls=ls, color="gray", lw=0.8, label=rf"MPS-hard, $N={N}$")
            ax.set(xlabel=xl + r" (floor $-30$ = exactly few-body)")
            simple_ticks(ax)
        axs[0].set_ylabel(r"$\max_t S$ [nats]")
        for i, ax in enumerate(axs):
            ax.set_title(panel(i), loc="left")
        cbar = fig.colorbar(sc, ax=axs, label=r"non-classicality $\eta$")
        simple_ticks(cbar.ax)
        fig.legend(*axs[1].get_legend_handles_labels(), loc="outside lower center", ncols=5)
        fig.savefig("figs/map.pdf"); fig.savefig("figs/map.png", dpi=110)
        plt.close(fig)


SCALING = [("R1", 0.05), ("R6", 1.0), ("I1", 6.0), ("D3", 0.0)]  # (family, g): N = 12 (sq4x3) vs N = 16 (sq4x4), init |+x>


def scaling():
    """Half-cut entropy per cut boundary bond, s = S/n_cut, at N = 12 and 16 on the square lattice."""
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, len(SCALING), figsize=(9.5, 2.6), sharey=True, layout="constrained")
        for i, (ax, (ID, g)) in enumerate(zip(axs, SCALING)):
            for lat, color in (("sq4x4", COLORS[0]), ("sq4x3", COLORS[1])):
                d = np.load(f"data/screen/{ID}_{lat}_+x.npz")
                k = int(np.argmin(np.abs(d["g"] - g)))
                N, ncut = int(d["N"][k]), int(d["ncut"][k])
                t = d["t_g"][k]
                ax.plot(t, d["S"][k] / ncut, color=color, lw=1, label=rf"$N={N}$ ({lat}, $n_\partial={ncut}$)")
                page = np.nanmax(d["S"][k] / d["S_rel"][k]) / ncut
                ax.axhline(page, ls=":", color=color, lw=0.8, label=rf"random state, $N={N}$")
            ax.set(xlabel=rf"time $t$ [1/{unit_label(str(d['unit'][0]))[2:]}]", xscale="log", xlim=(t[1], t[-1] if np.isfinite(t[-1]) else None))
            ax.set_title(panel(i) + rf" {ID}, {d['glabel']}$={g:g}$", loc="left")
            simple_ticks(ax)
        axs[0].set_ylabel(r"$S/n_\partial$ [nats per cut bond]")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
        fig.savefig("figs/scaling.pdf"); fig.savefig("figs/scaling.png", dpi=110)
        plt.close(fig)


MAIN = [("R6", "sq4x3", "+x"), ("R1", "sq4x3", "+x"), ("I1", "sq4x3", "+x"), ("R4", "kag12", "th"),
        ("D3", "sq4x3", "+x"), ("G4", "sq4x3", "gs")]  # Stage-3 families, Tier 1, near-classical first
HIGHLIGHT = [m for m in MAIN if m[0] in ("R1", "G4")]  # discussed in the main text
MODEL = {  # Hamiltonian and initial state drawn into the main-rows heatmaps; n = (1 - Z)/2, Pauli X, Y, Z
    "R6": r"$H=(\Omega/2)\sum X+\sum U_{ij}n_in_j$" "\n" r"$U_{ij}\propto[1+(r_{ij}/R_c)^6]^{-1}$" "\n"
          r"$\Omega=0.05\,U_{nn}$, from $|{+}x\rangle$",
    "R1": r"$H=(\Omega/2)\sum X+\sum V_{nn}r_{ij}^{-6}n_in_j$" "\n" r"from $|{+}x\rangle$",
    "I1": r"$H=\sum r_{ij}^{-\kappa}Z_iZ_j+0.1\sum X$" "\n" r"from $|{+}x\rangle$",
    "R4": r"$H=(\Omega/2)\sum X+\sum \Omega R_b^6r_{ij}^{-6}n_in_j$" "\n"
          r"from $\otimes(\cos0.1|g\rangle+\sin0.1|r\rangle)$",
    "D3": r"$H=\sum r_{ij}^{-3}(XX+YY+\Delta ZZ)/2$" "\n" r"$\Delta=J_z/J_{xy}$, from $|{+}x\rangle$",
    "G4": r"$H=-\sum r_{ij}^{-6}Z_iZ_j-h_f\sum X$" "\n" r"from gs($h_f=4$)",
    "S1": r"$H=-\sum A_s-\sum B_p+\epsilon\sum Z$" "\n" r"from $|\theta=\pi/4\rangle$",
    "I5": r"$H=\sum_\triangle ZZZ+h_x\sum X$" "\n" r"from $|{+}x\rangle$",
    "B1": r"$H=-\sum A_s-\sum B_p-h_f\sum Z$" "\n" r"from gs($h_f=0.1$)",
    "B2": r"$H=-\sum A_s-\sum B_p$" "\n" r"$-h_f\sum(Z+X)/\sqrt{2}$" "\n" r"from gs($h_f=0.1$)",
    "B3": r"$H=-\sum_\boxdot ZZZZ-h\sum X$", "B4": r"$H=-\sum_\triangle ZZZ-h\sum X$",
    "B5": r"$H=-\sum_{\triangle_\uparrow} ZZZ-h\sum X$",
    "B6": r"$H=-\sum ZZ-K\sum_\boxdot ZZZZ-0.5\sum X$" "\n" r"from $|{+}x\rangle$",
    "B7": r"$H=-\sum_\triangle ZZZ-\lambda\sum ZZ-0.5\sum X$" "\n" r"from $|{+}x\rangle$",
}
LATNAME = {"tc12": r"toric code $2\times3$", "tc18": r"toric code $3\times3$", "sq4x3": r"square $4\times3$",
           "sq4x4": r"square $4\times4$", "tri4x3": r"triangular $4\times3$", "tri4x4": r"triangular $4\times4$",
           "tri6x3": r"triangular $6\times3$"}
START = {"+x": r"from $|{+}x\rangle$", "gs": r"from gs($h=1.5$)"}  # B3-B5 come with two starts
B2B = [("B1", "tc18", "gs"), ("B1", "tc12", "gs"), ("B2", "tc18", "gs"), ("B2", "tc12", "gs"),
       ("B3", "sq4x4", "+x"), ("B3", "sq4x4", "gs"), ("B4", "tri6x3", "+x"), ("B4", "tri6x3", "gs"),
       ("B5", "tri4x4", "+x"), ("B5", "tri4x4", "gs"), ("B6", "sq4x4", "+x"), ("B7", "tri4x4", "+x"),
       ("G4", "sq4x3", "gs"), ("R1", "sq4x3", "+x"), ("S1", "tc12", "th"), ("I5", "tri4x3", "+x")]  # plan_beyond2body.md
lg = lambda x: np.log10(np.maximum(x, 1e-16))
PANELS = [("mu", r"$\mu$", MU_LIM, "viridis"), ("mu_ph", r"$\mu_{\rm ph}$", MU_LIM, "viridis"),
          ("S", r"$S$ [nats]", (0, 6 * np.log(2)), "viridis"), ("inf", r"$\log_{10}I$", (-6, 0), "magma")]
B2B_PANELS = [("rate2", r"$\log_{10}R_2$", (-3, 1), "viridis"), ("Phi2", r"$\log_{10}\Phi_2$", (-3, 3), "viridis"),
              ("w_hi", r"$\log_{10}w_{hi}$", (-4, 0), "viridis"), ("mu", r"$\mu$", MU_LIM, "viridis"),
              ("S", r"$S$ [nats]", (0, 9 * np.log(2)), "viridis")]
B2B_APPX = [("mu_ph", r"$\mu_{\rm ph}$", MU_LIM, "viridis"), ("w_hi_ph", r"$\log_{10}w_{hi}^{\rm ph}$", (-4, 0), "viridis"),
            ("rate4", r"$\log_{10}R_4$", (-3, 1), "viridis"), ("Phi4", r"$\log_{10}\Phi_4$", (-3, 3), "viridis"),
            ("chi_frac", r"$\chi_\epsilon/\chi_{\rm cap}$", (0, 1), "viridis")]
# onsets: (label, quantity, threshold, line style, colour); threshold may depend on N
ONSETS = [(r"$t^*_\mu$", "mu", lambda N: -10, "-", "white"), (r"$t^*_{\rm ph}$", "mu_ph", lambda N: -10, ":", "white")]
B2B_ONSETS = [(r"$t^*_\mu$ ($\mu$ or $\mu_{\rm ph}=-10$)", ("mu", "mu_ph"), lambda N: -10, "-", "white"),
              (r"$t^*_w$ ($w_{hi}$ or $w_{hi}^{\rm ph}=1\%$)", ("w_hi", "w_hi_ph"), lambda N: 0.01, ":", "white"),
              (r"$t_S$ ($S=S_{\rm Page}/2$)", "S", lambda N: S_RANDOM(N) / 2, "-.", "orange")]


def raw(d, name, k, n):
    """Row k (first n times) of a stored quantity; mu floored at FLOOR, w_hi = 0 where log|psi| is flat."""
    z = d[name][k, :n].astype(float)
    if name.startswith("mu"):
        return np.maximum(z, FLOOR)
    if name.startswith("w_hi"):  # a ratio of round-off (total weight ~1e-30) for a product state or Omega = 0
        W = d["W_ph" if name.endswith("ph") else "W"][k, :n]
        z = np.where(np.nansum(W, 1) > 1e-20, z, 0.0)
    return z


def quantity(d, name, k, n):
    """Row k (first n times) of a stored or derived quantity, in the units the panels show."""
    if name in ("Phi2", "Phi4"):  # accumulated few-body error int_0^t R_k dt'
        from scipy.integrate import cumulative_trapezoid
        return lg(cumulative_trapezoid(d["rate" + name[-1]][k, :n], d["t_g"][k, :n], initial=0))
    z = raw(d, name, k, n)
    return lg(z) if name.startswith(("rate", "w_hi")) else z


def first_cross(t, z, thr):
    return t[np.argmax(z > thr)] if (z > thr).any() else np.nan


def main_rows(fams=MAIN, out="main_rows", panels=PANELS, onsets=ONSETS, root="data/screen", rbm="data/rbm_long.csv"):
    """One row per family, one column per panel over (g, t); the 'inf' panel is the alpha = 1 RBM infidelity (rbm csv).
    Onset times (first threshold crossing of each predictor, and of I = 1e-3 if rbm is given) are drawn on every panel."""
    if rbm is not None:
        rbm = pd.read_csv(rbm)
        rbm = rbm[rbm.ansatz == "rbm1"]
        onsets = onsets + [(r"$t^*_{\rm inf}$", "inf", lambda N: 1e-3, "--", "cyan")]
    nc = len(panels)
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(len(fams), nc, figsize=(2.75 * nc, 2.15 * len(fams)), layout="constrained", squeeze=False)
        for r, (ID, lat, init) in enumerate(fams):
            d = np.load(f"{root}/{ID}_{lat}_{init}.npz")
            g, N = d["g"], int(d["N"][0])
            order = np.argsort(g)
            scaled = np.ptp(np.nanmax(d["t_g"], axis=1)) > 1e-9
            sub = rbm[(rbm.ID == ID) & (rbm.lat == lat) & (rbm.init == init)] if rbm is not None else None
            run_of = lambda k: sub[np.isclose(sub.g, g[k], atol=1e-3)].sort_values("t")
            tstar = {lab: [] for lab, *_ in onsets}
            for k in order:
                tk = d["t_g"][k][np.isfinite(d["t_g"][k])]
                for lab, name, thr, *_ in onsets:
                    if name == "inf":
                        run = run_of(k)
                        tstar[lab].append(first_cross(run.t.values, run.inf.values, thr(N)) if len(run) else np.nan)
                    else:  # a tuple of names: the first of their crossings
                        names = name if isinstance(name, tuple) else (name,)
                        tstar[lab].append(np.nanmin([first_cross(tk, raw(d, x, k, len(tk)), thr(N)) for x in names] + [np.inf]))
            for c, (name, label, lim, cmap) in enumerate(panels):
                ax = axs[r, c]
                for row, k in enumerate(order):
                    tk = d["t_g"][k][np.isfinite(d["t_g"][k])]
                    if name == "inf":
                        run = run_of(k)
                        z = np.full(len(tk), np.nan)
                        z[:len(run)] = np.log10(run.inf.clip(lower=1e-16).values)
                    else:
                        z = quantity(d, name, k, len(tk))
                    x = tk / tk[-1] if scaled else tk
                    edges = np.concatenate([[x[0]], (x[1:] + x[:-1]) / 2, [x[-1]]])
                    mesh = ax.pcolormesh(edges, [row - 0.5, row + 0.5], z[None, :], cmap=cmap, vmin=lim[0], vmax=lim[1])
                    if name == "S":  # chi_eps at its cap
                        capped = d["chi_frac"][k, :len(tk)] >= 0.999
                        ax.plot(x[capped], np.full(capped.sum(), row), ls="none", marker="s", ms=1.2, color="0.6")
                cbar = fig.colorbar(mesh, ax=ax, label=label, pad=0.01)
                simple_ticks(cbar.ax)
                ax.set_title(panel(nc * r + c), loc="left")
                unit = str(d["unit"][0])
                ax.set_xlabel(r"$t/t_{\max}(g)$" if scaled else rf"time $t$ [1/{unit if unit != '1' else '$J$'}]")
                simple_ticks(ax)
            # onset times on every panel: same rows, scaled like the data
            tend = np.array([np.nanmax(d["t_g"][k]) for k in order]) if scaled else 1.0
            for ax in axs[r]:
                rows = np.arange(len(g))
                for lab, _, _, ls, color in onsets:
                    ax.plot(np.array(tstar[lab]) / tend, rows, ls=ls, color=color, lw=1.1, label=lab)
                ax.set_yticks(rows, [f"${v:.3g}$" for v in g[order]], fontsize=6)  # categorical: grid points
                ax.set_ylim(-0.5, len(g) - 0.5)
            axs[r, 0].set_ylabel(f"{ID}: " + str(d["glabel"]).replace(r"\alpha", r"\kappa"))
            text = MODEL[ID] if root == "data/screen" else (
                MODEL[ID] + ("\n" + START[init] if ID in ("B3", "B4", "B5") else "") + f"\n{LATNAME.get(lat, lat)}, $N={N}$")
            model_label(axs[r, 0], text, lattice(lat) if not lat.startswith("tc") else None)
            for gm in d["marks_g"]:  # experimental / special values: red triangle at the left edge of every panel
                kk = int(np.argmin(np.abs(g[order] - float(gm))))
                for ax in axs[r]:
                    ax.plot(0, kk, marker=">", ms=5, color="red", clip_on=False, transform=ax.get_yaxis_transform(), zorder=5)
        handles = axs[0, -1].get_legend_handles_labels()
        handles[0].append(plt.Line2D([], [], ls="none", marker="s", ms=3, color="0.6")); handles[1].append(r"$\chi_\epsilon$ at cap")
        handles[0].append(plt.Line2D([], [], ls="none", marker=">", ms=5, color="red"))
        handles[1].append("experimental value" if root == "data/screen" else "special value")
        fig.legend(*handles, loc="outside lower center", ncols=5, facecolor="0.85")
        fig.savefig(f"figs/{out}.pdf"); fig.savefig(f"figs/{out}.png", dpi=100)
        plt.close(fig)


ANSATZ_LABEL = {"rbm1": r"RBM $\alpha=1$", "rbm2": r"RBM $\alpha=2$", "rbm4": r"RBM $\alpha=4$", "jas": "Jastrow",
                "rbmjas1": r"RBM $\alpha=1\times$Jastrow"}
ANSATZ_COLOR = dict(zip(ANSATZ_LABEL, COLORS))


def omega0():
    """Representability at Omega = 0 (R0 on the kagome cluster): infidelity per ansatz, and n_eff/N against alpha."""
    rbm = pd.read_csv("data/rbm_long.csv")
    r0 = rbm[(rbm.ID == "R0") & (rbm.lat == "kag12")]
    scr = np.load("data/screen/R0_kag12_th.npz")
    thetas = sorted(r0.g.unique())
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, len(thetas) + 1, figsize=(3.3 * (len(thetas) + 1), 2.8), layout="constrained")
        for i, (ax, th) in enumerate(zip(axs, thetas)):
            for a, run in r0[np.isclose(r0.g, th)].groupby("ansatz"):
                ax.plot(run.t, np.log10(run.inf.clip(lower=1e-16)), color=ANSATZ_COLOR[a], lw=1, label=ANSATZ_LABEL[a])
            ax.axhline(-3, ls="--", color="gray", lw=0.8, label=r"$I=10^{-3}$")
            ax.set(ylim=(-16.5, 0.5), xlabel=r"time $t$ [$1/V_{nn}$]", ylabel=r"$\log_{10}I$ (floor $-16$)")
            ax.set_title(panel(i) + rf" $\theta={th:.3g}$", loc="left")
            simple_ticks(ax)
            ax.legend(loc="lower right", fontsize=7)
        ax = axs[-1]
        k = 0  # the pair phases V_ij t/4 do not depend on the tilt: n_eff(t) is the same for every theta
        t = scr["t_g"][k]
        ax.plot(t, scr["n_eff"][k] / int(scr["N"][k]), color=COLORS[0], lw=1, label=r"$n_{\rm eff}/N$ (any $\theta$)")
        npairs = int(scr["N"][0]) * (int(scr["N"][0]) - 1) / 2 / int(scr["N"][0])
        ax.axhline(npairs, ls=":", color="gray", lw=0.8, label=r"all pairs, $(N-1)/2$")
        for al in (1, 2, 4):
            ax.axhline(al, ls="--", color=ANSATZ_COLOR[f"rbm{al}"], lw=0.8, label=rf"hidden units: $\alpha={al}$")
        ax.set(xlabel=r"time $t$ [$1/V_{nn}$]", ylabel=r"pairs per site", xscale="log", xlim=(t[1], t[-1]))
        ax.set_title(panel(len(thetas)) + r" two-body phases to encode", loc="left")
        simple_ticks(ax)
        ax.legend(loc="lower right", fontsize=7)
        fig.savefig("figs/omega0.pdf"); fig.savefig("figs/omega0.png", dpi=110)
        plt.close(fig)


ANSATZ_PTS = {"R1": [0.01, 0.05, 0.3], "R6": [0.5, 1.0, 2.02], "I1": [1.0, 3.0, 6.0], "R4": [1.15, 1.53, 2.4],
              "D3": [0.0, 0.75, 2.0], "G4": [0.304, 1.0, 3.044]}  # Stage-3 easy / onset / hard points


def ansatz(fams=MAIN, out="ansatz"):
    """log10 I(t) per ansatz at the three Stage-3 points of each family; N = 16 (alpha = 1) where it was run."""
    rbm = pd.read_csv("data/rbm_long.csv")
    tstar = pd.read_csv("data/rbm_tstar.csv")
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(len(fams), 3, figsize=(9, 1.9 * len(fams)), layout="constrained", squeeze=False)
        for r, (ID, lat, init) in enumerate(fams):
            for c, g in enumerate(ANSATZ_PTS[ID]):
                ax = axs[r, c]
                fam = rbm[(rbm.ID == ID) & np.isclose(rbm.g, g, atol=1e-3)]
                for (a, l), run in fam.groupby(["ansatz", "lat"]):
                    big = l != lat  # N = 16 repeat
                    ax.plot(run.t, np.log10(run.inf), color=ANSATZ_COLOR[a], lw=1, ls="--" if big else "-",
                            label=r"RBM $\alpha=1$, $N=16$" if big else ANSATZ_LABEL[a])
                t_S = tstar[(tstar.ID == ID) & (tstar.lat == lat) & np.isclose(tstar.g, g, atol=1e-3)].t_S.min()
                ax.axvline(t_S, color="gray", lw=0.8, ls=":", label=r"$t_S$: MPS-hard ($S=1.83$)")
                ax.axhline(-3, color="gray", lw=0.8, ls="--", label=r"$I=10^{-3}$")
                ax.set(ylim=(-10.5, 0.2), xscale="log", xlim=(fam.t[fam.t > 0].min(), fam.t.max()))  # log t: onset and late times
                ax.set_title(panel(3 * r + c) + rf" {ID}, $g={g:g}$", loc="left", fontsize=9)
                simple_ticks(ax)
            axs[r, 0].set_ylabel(r"$\log_{10}I$")
        for ax in axs[-1]:
            ax.set_xlabel(r"time $t$")
        labels = {}
        for ax in axs.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                labels.setdefault(l, h)
        fig.legend(labels.values(), labels.keys(), loc="outside lower center", ncols=4)
        fig.savefig(f"figs/{out}.pdf"); fig.savefig(f"figs/{out}.png", dpi=100)
        plt.close(fig)


B2B_COLORS = {"B4": COLORS[0], "B5": COLORS[1], "B2": COLORS[2], "B3": COLORS[3], "R1": COLORS[4]}  # one colour per family
WINDOW_ROWS = [("B4", "tri6x3", "+x"), ("B5", "tri4x4", "+x"), ("B2", "tc18", "gs"), ("B3", "sq4x4", "+x"), ("R1", "sq4x3", "+x")]
WINDOW_NAME = {"B4": r"Baxter-Wu, $|{+}x\rangle$", "B5": r"Newman-Moore, $|{+}x\rangle$", "B2": "toric code, diagonal field",
               "B3": r"Xu-Moore, $|{+}x\rangle$", "R1": r"Rydberg Ising (reference)"}


def b2b_windows(out="b2b/windows"):
    """Graphical abstract: fraction of the run that is MPS-hard and still predicted easy, against the control parameter g,
    for the mu criterion (a) and the w_hi criterion (b). Needs data/b2b/summary.csv (b2b_summary.py)."""
    df = pd.read_csv("data/b2b/summary.csv")
    criteria = [("win_mu", r"$\mu,\mu_{\rm ph}<-10$"), ("win_w", r"$w_{hi},w_{hi}^{\rm ph}<1\%$")]
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, layout="constrained")
        for i, (ax, (col, crit)) in enumerate(zip(axs, criteria)):
            for ID, lat, init in WINDOW_ROWS:
                sub = df[(df.ID == ID) & (df.lat == lat) & (df.init == init) & (df.g > 0) & df.quench].sort_values("g")
                ax.plot(sub.g, sub[col], "o-", ms=3, lw=1, color=B2B_COLORS[ID], label=WINDOW_NAME[ID])
            ax.set(xscale="log", xlabel=r"control parameter $g$ ($h$, $h_f$ or $\Omega/V_{nn}$)", ylim=(-0.03, 1.03))
            ax.set_title(panel(i) + " easy: " + crit, loc="left", fontsize=9)
            simple_ticks(ax)
        axs[0].set_ylabel("MPS-hard and predicted easy\n(fraction of save times)")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside right center", fontsize=8)
        fig.savefig(f"figs/{out}.pdf"); fig.savefig(f"figs/{out}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figs/screen", exist_ok=True)
    if sys.argv[1] == "ansatz":
        ansatz()
        ansatz(HIGHLIGHT, "ansatz_hl")
    if sys.argv[1] == "screen":
        for f in sorted(glob.glob("data/screen/*.npz")):
            sheet(f)
    elif sys.argv[1] == "map":
        overview_map()
    elif sys.argv[1] == "scaling":
        scaling()
    elif sys.argv[1] == "main":
        main_rows()
        main_rows(HIGHLIGHT, "main_rows_hl")
    elif sys.argv[1] == "omega0":
        omega0()
    elif sys.argv[1] == "b2b":  # plan_beyond2body.md: full matrix and appendix matrix; highlights via `b2b ID:lat:init ...`
        os.makedirs("figs/b2b", exist_ok=True)
        fams = [tuple(a.split(":")) for a in sys.argv[2:]] or B2B
        tag = "_hl" if sys.argv[2:] else ""
        if not tag:
            b2b_windows()
        for i in range(0, len(fams), 4):  # 4 rows x 5 panels per page: panel letters run to (t)
            main_rows(fams[i:i + 4], f"b2b/main{tag}_{i // 4 + 1}", B2B_PANELS, B2B_ONSETS, "data/b2b", None)
            main_rows(fams[i:i + 4], f"b2b/appx{tag}_{i // 4 + 1}", B2B_APPX, B2B_ONSETS, "data/b2b", None)
