"""Steps 3 and 6 of PLAN_quench_family.md: t* definitions, RBM-vs-proxy correlations.

Reads data/ed_*.npz and data/rbm/*.npz, writes
  data/table_long.csv     one row per (fam, p, t): ED proxies and RBM infidelities
  data/table_tstar.csv    one row per (fam, p): t*_mu, t_dev, t*_w, t*_inf, final values
  data/table_tstar_plane.csv  t*_mu, t_dev over the (Jxx, h_f) plane of family B
  data/correlations.csv   Spearman and partial Spearman of each proxy against log10 infidelity
"""
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

INF_THRESHOLD = 1e-3
PROXIES = ["mu", "mu_ph", "mu2", "T", "w_hi", "minlog", "S"]


def first_time(t, mask):
    """First t where mask holds, inf if never."""
    return t[np.argmax(mask)] if mask.any() else np.inf


def ed_table(name, fam):
    d = np.load(f"data/ed_{name}.npz")
    rows = []
    for i, p in enumerate(d["vals"]):
        df = pd.DataFrame({k: d[k][i] for k in PROXIES + ["R2"]})
        df["t"], df["fam"], df["p"] = d["times"], fam, round(float(p), 3)
        rows.append(df)
    return pd.concat(rows)


def rbm_table():
    rows = []
    for f in sorted(glob.glob("data/rbm/rbm_*.npz")):
        d = np.load(f)
        rows.append(pd.DataFrame({"t": d["times"], "inf": d["inf"], "fam": str(d["fam"]),
                                  "p": round(float(d["val"]), 3), "alpha": int(d["alpha"])}))
    df = pd.concat(rows)
    return df.pivot_table(index=["fam", "p", "t"], columns="alpha", values="inf").add_prefix("inf_a").reset_index()


def tstar_table(long):
    rows = []
    for (fam, p), g in long.groupby(["fam", "p"]):
        t, mu = g["t"].values, g["mu"].values
        row = dict(fam=fam, p=p, mu0=mu[0],
                   t_mu=first_time(t, mu > mu[0] + 5), t_dev=first_time(t, mu > mu[0] + 1.5),
                   t_w=first_time(t, g["w_hi"].values > 0.01),
                   mu_final=mu[-1], S_final=g["S"].values[-1], S_max=g["S"].max())
        if g["inf_a1"].notna().any():
            row["t_inf"] = first_time(t, g["inf_a1"].values > INF_THRESHOLD)
            row["inf_final"] = g["inf_a1"].values[-1]
            row["inf_max"] = g["inf_a1"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def plane_table():
    d = np.load("data/ed_plane.npz")
    t = d["times"]
    rows = [dict(p=round(float(p), 3), hf=float(hf), mu0=mu[0], t_mu=first_time(t, mu > mu[0] + 5),
                 t_dev=first_time(t, mu > mu[0] + 1.5), mu_final=mu[-1], S_final=S[-1])
            for p, hf, mu, S in zip(d["vals"], d["hf"], d["mu"], d["S"])]
    return pd.DataFrame(rows)


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after linearly regressing the ranks of `controls` out of both."""
    rx, ry = rankdata(x), rankdata(y)
    if not controls:
        return np.corrcoef(rx, ry)[0, 1]
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
    return np.corrcoef(res(rx), res(ry))[0, 1]


def correlations(long):
    rows = []
    sub = long.dropna(subset=["inf_a1"])
    for fam, g in [("A", sub[sub.fam == "A"]), ("B", sub[sub.fam == "B"]), ("A+B", sub)]:
        g = g[g.t > 0]  # t = 0 is the same state for every p
        y = np.log10(g["inf_a1"].clip(lower=1e-12))
        for proxy in PROXIES:
            x = g[proxy].values
            ok = np.isfinite(x)
            row = dict(fam=fam, proxy=proxy, n=int(ok.sum()),
                       rho=partial_spearman(x[ok], y[ok], []),
                       rho_given_S=partial_spearman(x[ok], y[ok], [g["S"].values[ok]]) if proxy != "S" else np.nan,
                       rho_given_S_t=partial_spearman(x[ok], y[ok], [g["S"].values[ok], g["t"].values[ok]]) if proxy != "S" else np.nan,
                       rho_given_t=partial_spearman(x[ok], y[ok], [g["t"].values[ok]]),
                       rho_given_mu=partial_spearman(x[ok], y[ok], [g["mu"].values[ok]]) if proxy != "mu" else np.nan,
                       rho_with_S=partial_spearman(x[ok], g["S"].values[ok], []))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    long = pd.concat([ed_table("B", "B"), ed_table("A", "A")])
    long["t"] = long["t"].round(10)
    rbm = rbm_table()
    rbm["t"] = rbm["t"].round(10)
    long = long.merge(rbm, on=["fam", "p", "t"], how="left")
    long.to_csv("data/table_long.csv", index=False)

    ts = tstar_table(long)
    ts.to_csv("data/table_tstar.csv", index=False)
    plane_table().to_csv("data/table_tstar_plane.csv", index=False)
    corr = correlations(long)
    corr.to_csv("data/correlations.csv", index=False)

    pd.set_option("display.width", 200)
    print(ts.round(3).to_string())
    print(corr.round(3).to_string())
    # success criteria of the plan
    for fam, g in ts.dropna(subset=["t_inf"]).groupby("fam"):
        cens = lambda s: s.replace(np.inf, 10.0)  # 'never' ranks above every finite time
        print(f"{fam}: Spearman(t*_mu, t*_inf) over p = {spearmanr(cens(g.t_mu), cens(g.t_inf))[0]:.3f}, "
              f"(t_w, t_inf) = {spearmanr(cens(g.t_w), cens(g.t_inf))[0]:.3f}")
    never = ts[np.isinf(ts.t_mu) & ts.inf_final.notna()]
    print("never-hard points (t*_mu = inf):\n", never[["fam", "p", "inf_final", "inf_max"]].to_string())
