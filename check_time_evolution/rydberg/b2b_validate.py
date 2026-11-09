"""Optional check of plan_beyond2body.md: do the predictors track the existing alpha = 1 RBM infidelity I (data/rbm_long.csv)
for the reference rows with RBM data (G4 and R1 on sq4x3)? -> data/b2b/validate_rho.csv, data/b2b/validate_onsets.csv

rho: Spearman correlation over all (g, t) with t > 0 between log10 I and each predictor.
onsets: t_inf (first I > 1e-3) against t_mu and t_w as in b2b_summary.py."""
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import spearmanr
from rydberg_figs import raw

rbm = pd.read_csv("data/rbm_long.csv")
rbm = rbm[rbm.ansatz == "rbm1"]
summ = pd.read_csv("data/b2b/summary.csv")
rows, onsets = [], []
for ID in ("G4", "R1"):
    d = np.load(f"data/b2b/{ID}_sq4x3_{'gs' if ID == 'G4' else '+x'}.npz")
    pts = []
    for k, g in enumerate(d["g"]):
        run = rbm[(rbm.ID == ID) & (rbm.lat == "sq4x3") & np.isclose(rbm.g, g, atol=1e-3)].sort_values("t")
        if run.empty:
            continue
        t = d["t_g"][k][np.isfinite(d["t_g"][k])]
        n = len(t)
        pred = pd.DataFrame(dict(t=t, R2=d["rate2"][k, :n], Phi2=cumulative_trapezoid(d["rate2"][k, :n], t, initial=0),
                                 w_hi=np.maximum(raw(d, "w_hi", k, n), raw(d, "w_hi_ph", k, n)),
                                 mu=np.maximum(raw(d, "mu", k, n), raw(d, "mu_ph", k, n))))
        m = pd.merge_asof(run[["t", "inf"]], pred, on="t", direction="nearest", tolerance=1e-6).dropna()
        pts.append(m[m.t > 0].assign(g=g))
        s = summ[(summ.ID == ID) & np.isclose(summ.g, g)].iloc[0]
        t_inf = run.t.values[np.argmax(run.inf.values > 1e-3)] if (run.inf > 1e-3).any() else np.inf
        onsets.append(dict(ID=ID, g=g, t_inf=t_inf, t_mu=s.t_mu, t_w=s.t_w, t_S=s.t_S))
    pts = pd.concat(pts)
    for col in ("R2", "Phi2", "w_hi", "mu"):
        rows.append(dict(ID=ID, predictor=col, rho=spearmanr(np.log10(pts.inf.clip(lower=1e-16)), pts[col]).statistic,
                         npts=len(pts)))
rho, ons = pd.DataFrame(rows), pd.DataFrame(onsets)
rho.to_csv("data/b2b/validate_rho.csv", index=False)
ons.to_csv("data/b2b/validate_onsets.csv", index=False)
if __name__ == "__main__":
    print(rho.round(2).to_string())
    print(ons.round(2).to_string())
