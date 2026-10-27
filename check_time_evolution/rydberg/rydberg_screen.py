"""Stage 1 ED screen of PLAN_rydberg_quench.md: family catalogue + parallel runner.

usage: python rydberg_screen.py run [TIER ...]      -> data/screen/raw/<ID>_<lat>_<init>_<k>.npz (resumable)
       python rydberg_screen.py gather               -> data/screen/<ID>_<lat>_<init>.npz
       python rydberg_screen.py dtcheck              -> dt/2 reruns at 3 g values per Tier-1 family (raw/dthalf_*)
"""
import os, sys, glob, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
from rydberg2d import *
from rydberg2d import _lat

PI = np.pi
RAW = "data/screen/raw"
lin = lambda a, b, n: list(np.round(np.linspace(a, b, n), 4))
tmax_field = lambda h: min(300.0, max(10.0, 3 / h)) if h > 0 else 300.0

# ID: tier, g label, g grid, lattices, inits, experimental marks {value: label}
FAMILIES = {
    # ---- Tier 1
    "S3":  (1, r"$h$", lin(0, PI / 4, 14), ["chain12", "chain16", "sq4x3"], ["+x", "th", "g~"], {}),
    "S3b": (1, r"$b=\theta_h/2$", lin(0, PI / 4, 14), ["ring12", "chain12", "sq4x3"], ["g~"],
            {0.0: "Clifford", PI / 4: "Clifford (Kim23)"}),
    "R0":  (1, r"$\theta$", lin(0.1, PI / 2, 14), ["chain16", "sq4x4", "tri4x4", "kag12"], ["th"], {PI / 2: "Zeiher16/17"}),
    "R6":  (1, r"$R_c/a$", [0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 1.75, 2.02, 2.25, 2.5, 2.75, 3.0], ["sq4x4", "tri4x4", "sq4x3"], ["+x"],
            {2.02: "Zeiher16", 3.0: "Zeiher16", 1.0: "Zeiher17"}),
    "R1":  (1, r"$\Omega/V_{nn}$", [0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.152, 0.2, 0.3, 0.4, 0.5, 1.0],
            ["sq4x3", "sq4x4", "tri4x4", "kag12"], ["+x"], {0.152: "SchmittHeyl20 $h_c/10$", 1.0: "Lienhard18"}),
    "G4":  (1, r"$h_f$", [0, 0.304, 0.6, 1.0, 1.5, 2.0, 2.5, 3.044, 3.5, 4.0, 4.5, 5.0, 6.089], ["sq4x3"], ["gs"],
            {0.304: "$h_c/10$", 3.044: "$h_c$", 6.089: "$2h_c$ (SchmittHeyl20)"}),
    "R4":  (1, r"$R_b/a$", [0.8, 1.0, 1.15, 1.3, 1.41, 1.53, 1.7, 1.85, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6], ["sq4x4", "kag12", "ruby12"],
            ["th", "Z2th"], {1.15: "Ebadi21", 1.53: "Semeghini21 (quench)", 2.4: "Semeghini21"}),
    "G1":  (1, r"$\Delta_f/\Omega$", lin(-1, 5, 14), ["sq4x4", "kag12"], ["gs"], {}),
    "R5":  (1, r"$\Delta/\Omega$", lin(-2, 2, 13), ["chain16", "sq4x4"], ["Z2th", "th"], {0.0: "Bernien17"}),
    "C2":  (1, r"$\omega_m/\Omega$", [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.24, 1.35, 1.5, 1.65, 1.8, 2.0], ["chain16"], ["Z2th"],
            {1.24: "Bluvstein21"}),
    "D1":  (1, r"$\Theta$", lin(0, PI / 2, 14), ["sq4x4", "tri4x4"], ["+x"], {0.0: "Bornet23"}),
    "G3":  (1, r"$J_z/J_{xy}$", lin(-1, 1, 15), ["sq4x4"], ["gs"], {0.0: "Chen23 (XY)"}),
    "D3":  (1, r"$J_z/J_{xy}$", lin(-2, 2, 17), ["sq4x4", "chain16", "sq4x3"], ["+x", "Z2th"], {1.0: "Scholl22 (Heisenberg)", 0.0: "XY"}),
    "D4":  (1, r"$\delta/J$", lin(0, 3, 13), ["kag12", "tri4x4"], ["+x", "Z2th", "120"], {0.0: "Bintz24 DSL Hamiltonian"}),
    "I1":  (1, r"$\kappa$", [0.5, 0.63, 0.83, 1.0, 1.08, 1.19, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0], ["chain16", "sq4x4", "sq4x3"], ["+x"],
            {0.63: "Richerme14", 0.83: "R14", 1.0: "R14/Zhang17", 1.19: "R14", 3.0: "dipolar", 6.0: "vdW"}),
    "Q1":  (1, r"$\beta$", lin(0, PI / 4, 14), ["sq4x4", "kag12"], ["+x"], {}),
    # ---- Tier 2 (main lattice, 8 points)
    "R3":  (2, r"$\Delta/V_{nn}$", [-1, 0, 0.5, 1, 1.5, 2, 2.5, 3], ["sq4x3"], ["+x", "th"], {}),
    "C1":  (2, r"$\delta$", lin(-0.3, 0.3, 8), ["chain12"], ["seed", "+x"], {}),
    "S1":  (2, r"$\epsilon$", lin(0, 0.5, 8), ["tc12"], ["th"], {}),
    "S2":  (2, r"$\epsilon$", lin(0, 0.5, 8), ["sq4x3"], ["th"], {}),
    "S4":  (2, r"$\epsilon$", lin(0, 0.5, 8), ["chain12"], ["+x"], {}),
    "I5":  (2, r"$h_x$", lin(0, 0.5, 8), ["tri4x3"], ["+x"], {}),
    "R9":  (2, r"$V_{\rm intra}/V_{\rm inter}$", lin(0, 1, 8), ["sq4x3"], ["+x"], {}),
    "R8":  (2, r"$\Omega$", [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], ["sq4x3"], ["+x"], {}),
    "I2":  (2, r"$h_x$", lin(0, 1, 8), ["sq4x3"], ["+x"], {}),
    "D2":  (2, r"$h_x$", lin(0, 1, 8), ["sq4x3"], ["+x"], {}),
    "D5":  (2, r"$\lambda$", lin(0, 3, 8), ["sq4x3"], ["+x"], {}),
    "G2":  (2, r"$\Omega_f$", [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], ["sq4x3"], ["gs"], {}),
    "R2":  (2, r"$\Omega$", [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], ["sq4x3"], ["th", "Z2th"], {}),
    "D6":  (2, r"filling $f$", [0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 1.0], ["sq4x4"], ["+x"], {}),
    "R7":  (2, r"$\sigma/a$", lin(0, 0.2, 8), ["sq4x3"], ["+x"], {}),
    "I3":  (2, r"$h_z$", lin(0, 1, 8), ["chain12"], ["th"], {}),
    # ---- Tier 3 (controls, 6 points)
    "I4":  (3, r"$h_x$", lin(0, 1, 6), ["sq4x3"], ["th"], {}),
    "S5":  (3, r"N\'eel tilt $\theta$", lin(0.1, PI / 2, 6), ["chain12"], ["Z2th"], {}),
    "S6":  (3, r"$\Omega$", lin(0, 0.5, 6), ["sq4x3"], ["th"], {}),
    "S7":  (3, r"$\Delta_{zz}$", lin(0, 1, 6), ["chain12"], ["Z2th"], {}),
    "C3":  (3, r"$s$", lin(-1, 1, 6), ["chain12"], ["th"], {}),
}
THETA = {"th": PI / 4, "Z2th": PI / 4, "g~": 0.1, "seed": 0.2}
THETA_SMALL = {"R4": 0.2, "R5": 0.2, "C2": 0.2, "I3": 0.1}  # near-|g> starts (Rydberg-native, tilted off the zeros)


def pairs(lat, f):
    """N x N matrix f(r, D) over i != j (min image)."""
    R, D = lat["R"], lat["D"]
    M = np.zeros_like(R)
    for i in range(lat["N"]):
        for j in range(lat["N"]):
            if i != j:
                M[i, j] = f(R[i, j], D[i, j])
    return M


def dipolar_xy(lat, J):
    """sum_{i<j} J_ij (XX + YY)/2 for matrix J."""
    return pair_terms(lat, lambda i, j: J[i, j] / 2, ("XX", "YY"))


def spec(ID, latname, init, g):
    """-> segs [(duration, terms)], psi0, t_save, period, units label."""
    lat = lattice(latname)
    N = lat["N"]
    vdw = pairs(lat, lambda r, d: r ** -6.0)
    nnJ = lambda i, j: 1.0 if (i, j) in lat["nn"] else 0.0
    th = THETA_SMALL.get(ID, THETA.get(init))
    psi0 = None if init == "gs" else initial_state(lat, "th" if init == "g~" else init, th if init != "th" or ID != "R0" else g)
    T201 = lambda tm: np.linspace(0, tm, 201)
    per, unit = None, r"$V_{nn}$"
    if ID == "R0":
        segs, ts = [(1, rydberg_terms(lat, 0, 0, vdw))], T201(100)
    elif ID in ("R1", "R2", "R8"):
        De = np.random.default_rng(1).uniform(-2, 2, N) if ID == "R8" else 0
        segs, ts = [(1, rydberg_terms(lat, g, De, vdw))], T201(tmax_field(g))
    elif ID == "R3":
        segs, ts = [(1, rydberg_terms(lat, 0.1, g, vdw))], T201(30)
    elif ID == "R4":
        segs, ts, unit = [(1, rydberg_terms(lat, 1.0, 0, g ** 6 * vdw))], T201(10), r"$\Omega$"
    elif ID == "R5":
        Om = 1 / 12
        segs, ts = [(1, rydberg_terms(lat, Om, g * Om, vdw))], T201(25 / Om)
    elif ID == "R6":
        U = pairs(lat, lambda r, d: (1 + g ** -6) / (1 + (r / g) ** 6))
        segs, ts = [(1, rydberg_terms(lat, 0.05, 0, U))], T201(60)
    elif ID == "R7":  # one disorder realisation (seed 1); distances in units of the bare lattice spacing
        jl = _lat(lat["pos"] + g * np.random.default_rng(1).normal(size=lat["pos"].shape), [(4, 0)], scale=1.0)
        segs, ts = [(1, rydberg_terms(lat, 0.1, 0, pairs(jl, lambda r, d: r ** -6.0)))], T201(30)
    elif ID == "R9":
        V = vdw * np.where(np.equal.outer(lat["sub"], lat["sub"]), g, 1.0)
        segs, ts = [(1, rydberg_terms(lat, 0.1, 0, V))], T201(30)
    elif ID == "I1":
        J = pairs(lat, lambda r, d: r ** -g)
        segs, ts = [(1, pair_terms(lat, lambda i, j: J[i, j]) + field_terms(N, 0.1, "X"))], T201(30)
    elif ID == "I2":
        J = pairs(lat, lambda r, d: r ** -3.0)
        segs, ts = [(1, pair_terms(lat, lambda i, j: J[i, j]) + field_terms(N, g, "X"))], T201(tmax_field(g))
    elif ID == "I3":
        segs, ts = [(1, pair_terms(lat, lambda i, j: -nnJ(i, j)) + field_terms(N, -0.25, "X") + field_terms(N, -g, "Z"))], T201(30)
    elif ID == "I4":
        segs, ts = [(1, pair_terms(lat, nnJ) + field_terms(N, g, "X"))], T201(tmax_field(g))
    elif ID == "I5":
        segs = [(1, [(1.0, P(N, {a: "Z", b: "Z", c: "Z"})) for a, b, c in triangles(lat)] + field_terms(N, g, "X"))]
        ts = T201(tmax_field(g))
    elif ID in ("D1", "D2"):
        Th = g if ID == "D1" else 0.0
        J = pairs(lat, lambda r, d: -(1 - 3 * np.sin(Th) ** 2 * (d[0] / r) ** 2) / r ** 3)
        segs, ts = [(1, dipolar_xy(lat, J) + field_terms(N, g if ID == "D2" else 0, "X"))], T201(10)
        unit = r"$J_{nn}$"
    elif ID == "D3":
        J = pairs(lat, lambda r, d: r ** -3.0)
        segs = [(1, dipolar_xy(lat, J) + pair_terms(lat, lambda i, j: g * J[i, j] / 2))]
        ts, unit = T201(10), r"$J_{nn}$"
    elif ID == "D4":
        J = pairs(lat, lambda r, d: r ** -3.0)
        segs, ts, unit = [(1, dipolar_xy(lat, J) + field_terms(N, g * lat["sub"], "Z"))], T201(10), r"$J_{nn}$"
    elif ID == "D5":
        segs = [(1, dipolar_xy(lat, pairs(lat, lambda r, d: r ** -3.0)) + pair_terms(lat, lambda i, j: g * vdw[i, j]))]
        ts, unit = T201(10), r"$J_{nn}$"
    elif ID == "D6":
        keep = np.sort(np.random.default_rng(1).permutation(N)[:int(round(g * N))])  # one realisation
        lat = _lat(lat["pos"][keep], [(4, 0), (0, 4)], scale=1.0)
        N = lat["N"]
        J = pairs(lat, lambda r, d: -r ** -3.0)
        psi0 = initial_state(lat, "+x")
        segs, ts, unit = [(1, dipolar_xy(lat, J))], T201(10), r"$J_{nn}$"
    elif ID == "C1":
        segs, ts = [(1, rydberg_terms(lat, 0.1, 1 + g, vdw))], T201(30)
    elif ID == "C2":
        Om, M = 1 / 12, 24
        Td = 2 * PI / (g * Om)
        tm = Td * (np.arange(M) + 0.5) / M
        segs = [(Td / M, rydberg_terms(lat, Om, Om * (0.85 + 0.98 * np.cos(g * Om * t)), vdw)) for t in tm]
        per, ts = Td, T201(25 / Om)
    elif ID == "C3":
        terms = field_terms(N, np.exp(-g), "X")[:1]
        for i in range(1, N):
            terms += [(np.exp(-g) / 2, P(N, {i: "X"})), (-np.exp(-g) / 2, P(N, {i - 1: "Z", i: "X"})), (0.5, P(N, {i - 1: "Z"}))]
        segs, ts, unit = [(1, terms)], T201(10), "1"
    elif ID == "S1":
        terms = ([(-1.0, P(N, {q: "X" for q in s})) for s in lat["stars"]] + [(-1.0, P(N, {q: "Z" for q in p})) for p in lat["plaq"]]
                 + field_terms(N, g, "Z"))
        segs, ts, unit = [(1, terms)], T201(10), "1"
    elif ID == "S2":
        nb = {i: [j for e in lat["nn"] for j in e if i in e and j != i] for i in range(N)}
        terms = [(-1.0, P(N, {i: "X", **{j: "Z" for j in nb[i]}})) for i in range(N)] + field_terms(N, g, "X")
        segs, ts, unit = [(1, terms)], T201(10), "1"
    elif ID in ("S3", "S3b", "S4"):
        J = PI / 4 + (g if ID == "S4" else 0)
        h = g if ID == "S3" else 0
        b = g if ID == "S3b" else PI / 4
        segs = [(1, pair_terms(lat, lambda i, j: J * nnJ(i, j)) + field_terms(N, h, "Z")), (1, field_terms(N, b, "X"))]
        per, ts, unit = 2.0, np.arange(0, 41, 2.0), "period/2"
    elif ID == "S5":
        psi0 = initial_state(lat, "Z2th", g)
        segs, ts, unit = [(1, pair_terms(lat, lambda i, j: nnJ(i, j) / 2, ("XX", "YY")))], T201(10), "1"
    elif ID == "S6":
        segs, ts, unit = [(1, pair_terms(lat, nnJ, ("XX",)) + field_terms(N, g, "Z"))], T201(tmax_field(g)), "1"
    elif ID == "S7":
        segs = [(1, pair_terms(lat, lambda i, j: nnJ(i, j) / 2, ("XX", "YY")) + pair_terms(lat, lambda i, j: g * nnJ(i, j) / 2))]
        ts, unit = T201(10), "1"
    elif ID == "G1":
        V = 1.15 ** 6 * vdw
        psi0 = ground_state(rydberg_terms(lat, 1.0, -2.0, V), N)
        segs, ts, unit = [(1, rydberg_terms(lat, 1.0, g, V))], T201(10), r"$\Omega$"
    elif ID == "G2":
        psi0 = ground_state(rydberg_terms(lat, 0.3, 3.0, vdw), N)
        segs, ts = [(1, rydberg_terms(lat, g, 3.0, vdw))], T201(tmax_field(g))
    elif ID == "G3":
        J = pairs(lat, lambda r, d: -r ** -3.0)
        psi0 = ground_state(dipolar_xy(lat, J) + field_terms(N, -0.2, "X"), N)
        segs, ts, unit = [(1, dipolar_xy(lat, J) + pair_terms(lat, lambda i, j: g * J[i, j] / 2))], T201(10), r"$J_{nn}$"
    elif ID == "G4":
        Hf = lambda h: pair_terms(lat, lambda i, j: -vdw[i, j]) + field_terms(N, -h, "X")
        psi0 = ground_state(Hf(4.0), N)
        segs, ts, unit = [(1, Hf(g))], T201(10), r"$J_{nn}$"
    elif ID == "Q1":
        U = pairs(lat, lambda r, d: 2.0 if r < 1.5 else 0.0)  # unit-disk (King's graph on the square lattice), MIS penalty 2
        segs = [(1, rydberg_terms(lat, 0, 1.0, U)), (1, field_terms(N, g, "X"))]  # gamma = 1 phase layer, beta = g mixer
        per, ts, unit = 2.0, np.arange(0, 17, 2.0), "layer/2"
    else:
        raise ValueError(ID)
    return dict(segs=segs, psi0=psi0, t_save=np.asarray(ts, float), period=per, N=N, unit=unit, lat=lat)


# ----------------------------------------------------------------------------- runner
def run_task(task):
    ID, latname, init, k, g, dt_scale, tag = task
    out = f"{RAW}/{tag}{ID}_{latname}_{init}_{k:02d}.npz"
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    s = spec(ID, latname, init, g)
    N, T = s["N"], sum(d for d, _ in s["segs"])
    parts = [(d, *sparse_parts(terms, N)) for d, terms in s["segs"]]
    rowsum = max(np.abs(o).sum(1).max() if o.nnz else 0.0 for _, _, o in parts)
    dts = np.diff(s["t_save"]).min()
    dt_max = dt_scale * (min(dts, 0.1 / rowsum) if rowsum > 0 else dts)
    t, ev, _ = evolve(s["psi0"], parts, s["t_save"], dt_max, period=s["period"])
    A = analyze(t, ev, N)
    dav = sum(d * dg for d, dg, _ in parts) / T
    oav = sum(d * o for d, _, o in parts) / T
    ncut = sum(1 for i, j in s["lat"]["nn"] if (i < N // 2) != (j < N // 2))
    np.savez(out, t=t, g=g, eta=eta(dav, sp.csr_matrix(oav)), N=N, ncut=ncut, dt_max=dt_max, unit=s["unit"],
             walltime=time.time() - t0, **A)
    return out, time.time() - t0


def tasks(tiers, dt_scale=1.0, tag=""):
    out = []
    for ID, (tier, _, grid, lats, inits, _) in FAMILIES.items():
        if tier in tiers:
            out += [(ID, l, i, k, g, dt_scale, tag) for l in lats for i in inits for k, g in enumerate(grid)]
    big = lambda t: ("16" in t[1] or t[1] == "sq4x4") and t[0] != "D6"
    return sorted(out, key=lambda t: (FAMILIES[t[0]][0], not big(t)))  # tier order, 16-site jobs first


def gather():
    groups = {}
    for f in sorted(glob.glob(f"{RAW}/*.npz")):
        key = os.path.basename(f).rsplit("_", 1)[0]
        groups.setdefault(key, []).append(f)
    for key, files in groups.items():
        ds = [dict(np.load(f)) for f in files]
        ID = key.split("_")[-3] if key.startswith("dthalf") else key.split("_")[0]
        nt = max(len(d["t"]) for d in ds)
        stack = {}
        for name in ds[0]:
            if np.ndim(ds[0][name]) == 0:
                stack[name] = np.array([d[name] for d in ds])
            else:
                shape = np.max([d[name].shape for d in ds], axis=0)  # W spectra can differ in length between g values
                stack[name] = np.array([np.pad(d[name].astype(float), [(0, n - m) for n, m in zip(shape, d[name].shape)],
                                               constant_values=np.nan) for d in ds])
        cap = np.round(stack["chi"] / np.where(stack["chi_frac"] > 0, stack["chi_frac"], np.nan))
        stack["chi_frac"] = (stack["chi"] - 1) / (cap - 1)  # 0 for a product state, 1 at the cap (raw files store chi/cap)
        tier, glabel, _, _, _, marks = FAMILIES[ID]
        np.savez(f"data/screen/{key}.npz", tier=tier, glabel=glabel, marks_g=list(marks), marks_l=list(marks.values()),
                 t=ds[int(np.argmax([len(d["t"]) for d in ds]))]["t"], t_g=stack.pop("t"), **stack)  # t_g: per-g times (t_max can depend on g)
    print(f"gathered {len(groups)} combinations")


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(RAW, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "gather":
        gather(); sys.exit()
    if cmd == "run":
        todo = tasks([int(x) for x in sys.argv[2:]] or [1, 2, 3])
    elif cmd == "dtcheck":
        todo = []
        for ID, (tier, _, grid, lats, inits, _) in FAMILIES.items():
            if tier == 1:
                todo += [(ID, lats[0], inits[0], k, grid[k], 0.5, "dthalf_") for k in (2, len(grid) // 2, len(grid) - 2)]
                todo += [(ID, lats[0], inits[0], k, grid[k], 1.0, "") for k in (2, len(grid) // 2, len(grid) - 2)]
    nproc = int(os.environ.get("NPROC", 28))
    print(f"{len(todo)} tasks on {nproc} processes", flush=True)
    with ProcessPoolExecutor(nproc) as ex:
        for i, (f, w) in enumerate(ex.map(run_task, todo)):
            print(f"[{i + 1}/{len(todo)}] {f} {w:.0f}s", flush=True)
