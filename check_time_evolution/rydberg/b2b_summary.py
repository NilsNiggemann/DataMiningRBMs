"""Onset times and the predicted easy-and-MPS-hard window per (family, g) of the beyond-two-body screen
(plan_beyond2body.md) -> data/b2b/summary.csv.

t_mu = first time max(mu, mu_ph) > -10, t_w = first time max(w_hi, w_hi_ph) > 1 %, t_S = first time S >= S_Page/2.
window_x: fraction of the save times at which the state is MPS-hard (S >= S_Page/2) and t < t_x, i.e. MPS-hard but still
predicted easy by x. S oscillates at weak fields, so this is less than the time after t_S."""
import glob, os
import numpy as np
import pandas as pd
from rydberg_figs import raw, first_cross, S_RANDOM

rows = []
for f in sorted(glob.glob("data/b2b/*.npz")):
    d = np.load(f)
    ID, lat, init = os.path.basename(f)[:-4].split("_")
    N = int(d["N"][0])
    for k in np.argsort(d["g"]):
        t = d["t_g"][k][np.isfinite(d["t_g"][k])]
        n = len(t)
        on = lambda name, thr: first_cross(t, raw(d, name, k, n), thr)
        t_mu = np.nanmin([on("mu", -10), on("mu_ph", -10), np.inf])
        t_w = np.nanmin([on("w_hi", 0.01), on("w_hi_ph", 0.01), np.inf])
        t_S = on("S", S_RANDOM(N) / 2)
        hard = d["S"][k, :n] >= S_RANDOM(N) / 2
        win = lambda te: float(np.mean(hard & (t < te)))
        h0 = {"B1": 0.1, "B2": 0.1, "G4": 4.0}.get(ID, 1.5) if init == "gs" else np.nan  # field of the start state
        rows.append(dict(ID=ID, lat=lat, init=init, N=N, g=d["g"][k], quench=not np.isclose(d["g"][k], h0), t_max=t[-1], eta=d["eta"][k], gap=d["gap"][k],
                         t_mu=t_mu, t_w=t_w, t_S=t_S, win_mu=win(t_mu), win_w=win(t_w), S0=d["S"][k, 0], S_max=d["S"][k].max(),
                         R2_0=d["rate2"][k, 0], R4_0=d["rate4"][k, 0], supp=int(d["supp"][k, -1]),
                         rank4=int(d["rank4"][k, -1])))
df = pd.DataFrame(rows)
df.to_csv("data/b2b/summary.csv", index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(df.drop(columns=["gap", "supp", "rank4", "R4_0"]).round(3).to_string())
