"""Stage-1 triage (PLAN_rydberg_quench.md step 3). Reads data/screen/*.npz, writes
  data/triage.csv          one row per (combination, g)
  data/triage_families.csv one row per combination: class counts and t* spread
"""
import glob, os
import numpy as np
import pandas as pd

THR = -10.0  # absolute thresholds for t*_mu and t*_ph (plan, carried over from the 4x3 study)
FLOOR = -30.0  # mu = -inf (exactly few-body) is shown at this value


def first_time(t, mask):
    return t[np.argmax(mask)] if mask.any() else np.inf


WINDOW = 2.0  # proxies stay below THR for >= WINDOW x the time S needs to reach 0.5 S_Page


def classify_window(r):
    """Time-window version of the plan's classes: MPS-hard (S_rel >= 0.5) long before either proxy fires."""
    if r.window >= WINDOW:
        return "gold" if r.eta >= 0.3 else ("near-classical" if r.eta <= 0.1 else "window (0.1<eta<0.3)")
    return classify(r) if classify(r) in ("onset", "boring") else "boring"


def classify(r):
    """Plan section 5.3, evaluated up to t_max."""
    hard_ph = np.isfinite(r.t_ph)
    if r.eta >= 0.3 and r.S_rel_max >= 0.5 and not np.isfinite(r.t_mu) and not hard_ph:
        return "gold"
    if r.eta <= 0.1 and r.S_rel_max >= 0.5 and not hard_ph:
        return "near-classical"
    if np.isfinite(r.t_mu) and r.mu_before < THR - 2:
        return "onset"
    return "boring"


rows = []
for f in sorted(glob.glob("data/screen/*.npz")):
    key = os.path.basename(f)[:-4]
    if key.startswith("dthalf"):
        continue
    d = np.load(f)
    ID, lat, init = key.split("_")
    for k, g in enumerate(d["g"]):
        t = d["t_g"][k][: np.isfinite(d["S"][k]).sum()]  # per-g time axis
        n = len(t)
        mu, muph, S = d["mu"][k, :n], d["mu_ph"][k, :n], d["S"][k, :n]
        mu_f = np.where(np.isnan(mu), FLOOR, np.maximum(mu, FLOOR))
        ph_f = np.where(np.isnan(muph), FLOOR, np.maximum(muph, FLOOR))
        t_mu, t_ph = first_time(t, mu_f > THR), first_time(t, ph_f > THR)
        t_S = first_time(t, d["S_rel"][k, :n] >= 0.5)  # MPS-hard (at this size) from here on
        early = t <= t[-1] / 5
        vE = np.max(np.gradient(S[early], t[early])) if early.sum() > 2 else np.nan
        rows.append(dict(ID=ID, tier=int(d["tier"]), lat=lat, init=init, g=float(g), N=int(d["N"][k]),
                         eta=float(d["eta"][k]), S_rel_end=d["S_rel"][k, n - 1], S_rel_max=np.nanmax(d["S_rel"][k, :n]),
                         vE=vE, mu_max=mu_f.max(), muph_max=ph_f.max(), mupht_max=np.nanmax(np.maximum(d["mu_pht"][k, :n], FLOOR)),
                         t_mu=t_mu, t_ph=t_ph, t_S=t_S, window=min(t_mu, t_ph) / t_S if np.isfinite(t_S) else 0.0, mu_before=mu_f[t <= t_mu / 2].max() if np.isfinite(t_mu) else mu_f.max(),
                         neff_frac=np.nanmax(d["n_eff"][k, :n]) / d["N"][k], chi_frac_end=d["chi_frac"][k, n - 1],
                         chi_frac_max=np.nanmax(d["chi_frac"][k, :n]), minlog=np.nanmin(d["minlog"][k, :n]),
                         w_hi_max=np.nanmax(d["w_hi"][k, :n]), t_end=t[-1], walltime=float(d["walltime"][k]),
                         mu_at_S=mu_f[t <= t_S].max(), muph_at_S=ph_f[t <= t_S].max()))  # proxies up to the MPS-hard time
df = pd.DataFrame(rows)
df["cls"] = df.apply(classify, axis=1)
df["cls_window"] = df.apply(classify_window, axis=1)
df.to_csv("data/triage.csv", index=False)

fam = (df.groupby(["ID", "tier", "lat", "init"])
       .apply(lambda g: pd.Series(dict(
           n=len(g), eta=g.eta.median(), S_rel_max=g.S_rel_max.max(),
           n_gold=(g.cls == "gold").sum(), n_nc=(g.cls == "near-classical").sum(), n_onset=(g.cls == "onset").sum(),
           n_gold_w=(g.cls_window == "gold").sum(), n_nc_w=(g.cls_window == "near-classical").sum(), win_max=g.window.max(),
           t_mu_min=g.t_mu.min(), t_mu_max=g.t_mu[np.isfinite(g.t_mu)].max() if np.isfinite(g.t_mu).any() else np.inf,
           n_never_mu=int(np.isinf(g.t_mu).sum()), n_never_ph=int(np.isinf(g.t_ph).sum()))), include_groups=False)
       .reset_index().sort_values(["tier", "ID"]))
fam.to_csv("data/triage_families.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(fam.round(3).to_string(index=False))
    print(df.cls.value_counts(), df.cls_window.value_counts())
    for c in ["gold", "near-classical"]:
        print(f"\n--- {c} (window)\n", df[df.cls_window == c][["ID", "lat", "init", "g", "eta", "t_S", "t_mu", "t_ph", "window", "neff_frac", "t_end"]]
              .round(3).to_string(index=False))
