Look a the report in @check_time_evolution/report/quench2d_report.tex

try to optimize local basis for the problem R1 for g=0.1 by minimizing mu via local basis rotations. These may be site dependent. The optimization may be not so be convex but not very expensive. Make sure to run the optimization with several initial guesses and different algorithms to ensure that a very good solutions is found.
mu can be computed as outlined in the report

Then, try to find a basis that minimizes mu for the entire quench time evolution. perhaps this basis rotation may also be time dependent.
Once this is found test whether the RBM infidelity improves.
write this in a new section in the report and compile the pdf.

# ============================================================================
# Detailed plan (2026-10-27, after Q&A)
# ============================================================================

Decisions from Q&A
- Objective J(t) = max(mu, mu_ph) of the rotated state (what t_prox = min(t_mu, t_ph) tracks).
- Step 1: a separate fixed-basis optimum at a few snapshot times.
- Step 2: first one fixed basis for the whole trajectory, then a smooth W(t).
- RBM test: both a supervised SR fit to the rotated exact states and full-sum TDVP in the rotated frame.

Setting (unchanged from the report)
- R1, sq4x3 (N=12), start |+x>, Omega/V_nn = g = 0.1, t in [0, 30], 201 save times; eta = small.
- Baseline (z basis): alpha=1 RBM t_inf = 1.65, final inf = 0.30 (data/rbm/R1_sq4x3_+x_+0.1000_rbm1.npz).
  Exact mu: -inf -> -9.0 (t=3.75) -> -5.8 (t=30); mu_ph: -inf -> -4.7; S ~ 3-3.8 nats.
- The raw screen file stores only the proxies, not psi(t), so we re-evolve with rydberg2d.evolve (seconds at N=12).

0. Rotation and objective (new file rydberg/basis_opt.py; reuse rydberg2d / quench2d.fwht, fit_mu)
- Rotated state psi' = W psi, W = prod_i W_i, W_i = R_y(theta_i) R_z(phi_i): 2 angles/site, 24 parameters.
  (A further R_z on the left only adds one-body phases, i.e. order-1 clusters, so it is dropped.)
  Apply W as 12 single-site 2x2 contractions on the (2,)*N tensor, batched over times.
- mu_ph needs the time-unwrapped phase, so for each W:
  rotate a fine trajectory (dense grid, about 2000 states, ~130 MB complex), then np.unwrap arg(W psi(t)) along t,
  subtract the mean, and fit at the objective times only.
- Same conventions as the report: noise cut log|c|<-28, A0 = N_real, mu = -inf if <=2-body (clipped at -30 in the
  objective), a NaN or runaway fit -> penalty +10.
- Sanity checks (asserts in __main__):
  W = I reproduces the stored mu(t), mu_ph(t);
  S(t) is invariant under W (local unitary);
  sum_s |psi'|^2 = 1.
- Note: at t=0 every node-free W keeps |+x> a product state, so mu = mu_ph = -inf. W(0) is therefore free.
  Rotations close to the x basis make amplitudes vanish (log|psi'| -> -inf), and the objective punishes them
  automatically. The 1e-300 floor from evolve() keeps it finite.

1. Fixed basis per snapshot (g = 0.1)
- Snapshot times: t = 1.65 (RBM failure), ~5, 30.
- Objective: J(t_k; W) for the fixed W applied along [0, t_k] (needed for the phase unwrapping).
- Parameter hierarchy, to see where the gain comes from and to seed the next level:
  (a) uniform W (2 params, a full grid scan, used as a map),
  (b) row-dependent (y = 0,1,2; x-translation of the cylinder; 6 params),
  (c) fully site-dependent (24 params).
- Starts: identity, the best grid points of (a), the best of (b) lifted to (c), sublattice-staggered angles,
  and >= 32 random starts.
- Algorithms (scipy only, no new dependency): Nelder-Mead, Powell, differential_evolution, dual_annealing,
  basinhopping(L-BFGS-B, finite differences). Optionally optuna TPE.
  The objective is non-smooth (sort, noise cut, curve_fit), so derivative-free methods are primary.
- Run starts in parallel with multiprocessing, <= 30 processes.
- Converged = the best J agrees across >= 3 algorithms/starts within 0.1.
  Report the spread of local minima, not just the best one.
- Output: data/basis/snap_t{t}.npz (angles, J, mu, mu_ph, w_hi per run).

2. Basis for the whole trajectory
- 2a, fixed W: minimise mean_t J(t) over ~25 times on [0, 30] (validate on all 201).
  Also report max_t J and t_prox. Same starts and algorithms as step 1, warm-started from the step-1 optima.
- 2b, smooth W(t): theta_i(t), phi_i(t) as cubic splines with ~6 knots (24 x 6 = 144 params), W(0) free.
  Objective: mean_t J(t) + lambda * int |dW/dt|^2 dt (lambda scanned, 2-3 values, so that the frame term stays
  small). Warm start: 2a, plus a spline through the step-1 snapshot optima.
- Lower bound for comparison: the step-1 per-time optima. If 2b gets near it, a time-dependent frame is enough.
- Output: data/basis/traj_fixed.npz, traj_spline_lam{lam}.npz; figures of mu, mu_ph(t) for z / 2a / 2b, plus the
  angles over time.

3. RBM test (alpha = 1; alpha = 2 if the gain is clear)
- 3a, expressivity (supervised): at ~10 times, SR infidelity minimisation of the RBM against psi(t)
  (z basis) and W psi(t) (2a, 2b). Same settings as the ground-state start in rydberg_rbm.py
  (SR diag_shift 5e-4, Sgd 0.01, 1500 steps), 2-3 seeds; report the best and the median infidelity.
  This separates expressivity from TDVP integration error, which the report suspects for R1.
- 3b, TDVP in the rotated frame:
  fixed W -> H' = W H W^dag. Conjugate each Pauli term: ZZ -> sum_ab R_ia R_jb s^a s^b, so 66 pairs x 9 strings.
    Build PauliStrings (Y is supported by sparse_parts/nk_operator).
    Check: dense H' == W H W^dag.
  time-dependent W(t) -> H'(t) = W H W^dag + i (dW/dt) W^dag. The extra term is one-body.
    Run it as piecewise-constant segments on a fine grid (midpoint W), with the existing segs machinery of
    rydberg_rbm.py.
    Check: the exact evolution under H'(t) equals W(t) psi(t).
  Initial RBM: the exact product start through the visible biases, as now (W|+x> is a product state, nodeless).
  Same solver settings as the baseline. Compare t_inf and inf(t) with 1.65 / 0.30.
- Minimal change: rydberg_rbm.py takes an optional basis file (angles / spline) and rotates the terms
  and psi_ex. No new runner.

4. Report
- New section in Part II after Sec. sec:r1, e.g. "R1 in an optimised local basis", written with the `report`
  skill (~/Projects/claude-skills/plugins/report). Contents:
  - the result first: the reduction of max(mu, mu_ph) and the change in t_inf;
  - figures with standalone captions;
  - a table (basis | mean J | t_prox | supervised inf at t = 1.65/5/30 | TDVP t_inf);
  - eta of H' (it goes up: the rotated frame is less classical);
  - limitations (non-convex, fit-based objective; one g; N = 12).
- Compile the pdf (latexmk -pdf quench2d_report.tex in report/).

Cost estimate (not measured)
- One J evaluation at a single time: a rotation plus 2 FWHTs plus 2 curve_fits, ~10-30 ms.
- A trajectory objective (25 times plus rotating the fine grid): ~1 s.
- Step 1 and 2a: minutes to ~1 h on 30 cores. Step 2b: a few hours.
- 3a: ~10 times x 3 bases x 3 seeds, minutes each. 3b: 3 TDVP runs, 5-30 min each.

Risks / premises to check
- The z basis may already be close to optimal at g = 0.1 (near-classical, eta small). A null result is possible,
  and is worth reporting.
- A lower mu does not imply a lower RBM infidelity: the RBM structure is basis-dependent. This is exactly the test.
  S is unchanged, which makes the comparison clean.
- mu_ph depends on the unwrapping path. It is defined consistently as the unwrapping of W(t') psi(t') over [0, t].
- Rotations break the structure of the z basis (all-to-all ZZ becomes general 2-body), so TDVP may get stiffer.
  Report the wall time.
