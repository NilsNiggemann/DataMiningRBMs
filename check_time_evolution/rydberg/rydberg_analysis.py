"""RBM vs proxies for the Rydberg study (PLAN_rydberg_quench.md step 6). Reads data/rbm/*.npz and the matching
screen files, writes
  data/rbm_long.csv      one row per (run, t): proxies, S, chi fraction, infidelity
  data/rbm_tstar.csv     one row per run: t*_mu, t*_ph, t_S, t*_inf and the MPS-hard / NQS-easy time
  data/rbm_corr.csv      Spearman and partial Spearman of each proxy against log10 infidelity, per family
"""
import glob, os

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quench2d"))  # quench2d.py, analysis.py, plotstyle.py
from analysis import partial_spearman, INF_THRESHOLD

THR, FLOOR = -10.0, -30.0
PROXIES = ["mu", "mu_ph", "S", "chi_frac"]


def first_time(t, mask):
    return t[np.argmax(mask)] if mask.any() else np.inf


def long_table():
    rows = []
    for f in sorted(glob.glob("data/rbm/*.npz")):
        r = np.load(f)
        ID, lat, init, ansatz, g = str(r["ID"]), str(r["lat"]), str(r["init"]), str(r["ansatz"]), float(r["g"])
        scr = np.load(f"data/screen/{ID}_{lat}_{init}.npz")
        k = int(np.argmin(np.abs(scr["g"] - g)))
        assert abs(scr["g"][k] - g) < 1e-3, (f, scr["g"])
        n = min(len(r["t"]), np.isfinite(scr["t_g"][k]).sum())
        assert np.allclose(r["t"][:n], scr["t_g"][k][:n])
        for i in range(n):
            if np.isnan(r["inf"][i]):
                break
            rows.append(dict(ID=ID, lat=lat, init=init, g=g, ansatz=ansatz, t=r["t"][i], inf=max(r["inf"][i], 1e-16),
                             eta=float(scr["eta"][k]), N=int(scr["N"][k]),
                             mu=max(scr["mu"][k, i], FLOOR), mu_ph=max(scr["mu_ph"][k, i], FLOOR),
                             S=scr["S"][k, i], S_rel=scr["S_rel"][k, i], chi_frac=scr["chi_frac"][k, i]))
    return pd.DataFrame(rows)


def tstar_table(long):
    rows = []
    for (ID, lat, init, g, ansatz), d in long.groupby(["ID", "lat", "init", "g", "ansatz"]):
        t = d.t.values
        easy = (d.S_rel.values >= 0.5) & (d.inf.values < INF_THRESHOLD)  # MPS-hard (at this N) and NQS-easy
        rows.append(dict(ID=ID, lat=lat, init=init, g=g, ansatz=ansatz, eta=d.eta.iloc[0], t_end=t[-1],
                         t_mu=first_time(t, d.mu.values > THR), t_ph=first_time(t, d.mu_ph.values > THR),
                         t_S=first_time(t, d.S_rel.values >= 0.5), t_inf=first_time(t, d.inf.values > INF_THRESHOLD),
                         inf_end=d.inf.values[-1], S_rel_max=d.S_rel.max(),
                         S_rel_easy=d.S_rel.values[easy].max() if easy.any() else np.nan,
                         t_easy=t[easy].max() if easy.any() else np.nan))
    return pd.DataFrame(rows)


def correlations(long):
    rows = []
    sub = long[(long.t > 0) & long.ansatz.isin(["rbm1"])]
    for fam, d in list(sub.groupby("ID")) + [("all", sub)]:
        y = np.log10(d.inf.values)
        for proxy in PROXIES:
            x = d[proxy].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5 or np.ptp(x[ok]) == 0:
                continue
            S, t = d.S.values[ok], d.t.values[ok]
            rows.append(dict(fam=fam, proxy=proxy, n=int(ok.sum()), n_g=d.g.nunique(),
                             rho=partial_spearman(x[ok], y[ok], []),
                             rho_given_S=partial_spearman(x[ok], y[ok], [S]) if proxy != "S" else np.nan,
                             rho_given_S_t=partial_spearman(x[ok], y[ok], [S, t]) if proxy != "S" else np.nan,
                             rho_given_mu=partial_spearman(x[ok], y[ok], [d.mu.values[ok]]) if proxy != "mu" else np.nan))
    return pd.DataFrame(rows)


def ordering(ts):
    """Per family (rbm1, N = 12): does the proxy onset order and precede the RBM failure time?"""
    from scipy.stats import spearmanr
    rows = []
    sub = ts[(ts.ansatz == "rbm1") & ~ts.lat.isin(["sq4x4"])].copy()
    sub["t_prox"] = np.minimum(sub.t_mu, sub.t_ph)
    for fam, d in sub.groupby("ID"):
        ok = np.isfinite(d.t_prox) & np.isfinite(d.t_inf)
        rows.append(dict(fam=fam, n_g=len(d), n_both=int(ok.sum()),
                         rho_mu=spearmanr(d.t_mu[ok], d.t_inf[ok])[0] if ok.sum() > 3 else np.nan,
                         rho_ph=spearmanr(d.t_ph[ok], d.t_inf[ok])[0] if ok.sum() > 3 else np.nan,
                         ratio_med=np.median(d.t_inf[ok] / d.t_prox[ok]) if ok.any() else np.nan,
                         frac_inf_first=float(np.mean(d.t_inf[ok] < d.t_prox[ok])) if ok.any() else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    long = long_table()
    long.to_csv("data/rbm_long.csv", index=False)
    ts = tstar_table(long)
    ts.to_csv("data/rbm_tstar.csv", index=False)
    corr = correlations(long)
    corr.to_csv("data/rbm_corr.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(ts.round(4).to_string(index=False))
    print(corr.round(3).to_string(index=False))
    order = ordering(ts)
    order.to_csv("data/rbm_order.csv", index=False)
    print(order.round(3).to_string(index=False))
    quad = long[(long.chi_frac >= 0.9) & (long.inf < INF_THRESHOLD)]
    print(f"\nMPS-hard / NQS-easy quadrant (chi fraction >= 0.9, I < {INF_THRESHOLD}): {len(quad)} of {len(long)} points")
    print(quad.groupby(["ID", "g", "ansatz"]).agg(n=("t", "size"), t_max=("t", "max"), eta=("eta", "first")).to_string())
