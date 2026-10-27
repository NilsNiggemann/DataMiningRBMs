# Plan: a one-parameter 2D quench family that is RBM-easy until t*, then hard

**Goal.** Find a physically meaningful 2D quench family, tuned by one control parameter p,
where the cumulant-convergence proxy μ(t) stays very negative up to t*(p) and then rises sharply.
Then show that RBM+TDVP infidelity follows μ qualitatively across (t, p).

Decisions already made (2026-10-26):
- **Comparison target:** full-sum RBM+TDVP infidelity vs ED only. No per-time supervised fits.
- **Proxy:** μ from log|ψ| is the main quantity. A phase cumulant μ_ph is a secondary diagnostic.
- **Lattice:** 4x3 only: periodic in x (length 4), open in y (length 3). PBC with length 3 would
  frustrate the bipartite structure, which makes the sign of Jxx ill-defined.
- **Basis:** RBM and cumulants both use the σ^z basis. Sites i = x·Ly + y, NetKet ordering.

Quick-test code and figures live in `quick_tests/`. They were run in the
`.venv` interpreter, CPU only.

---

## 1. Cumulant proxy: procedure (from `cumulants_time_evolution.ipynb`)

1. Compute the exact ψ(t) by ED. For 4x3, full `eigh` takes a few seconds. Use `evolve()` in `quick_tests/qt.py`.
2. Compute the cluster coefficients c_A(t) = WHT of log|ψ(s,t)|, i.e. (1/2^N) Σ_s χ_A(s) log|ψ(s,t)|.
   `qt.fwht` is a plain-numpy equivalent of `ce.fwht_coeffs_in_cluster_col_order`; the order of
   coefficients doesn't matter for the fit.
3. Drop c_∅. Drop symmetry-forced zeros, i.e. any log|c_A| < −28. Spin-flip Z2 makes all odd
   clusters vanish here.
4. Sort |c_A| in descending order. Fit rank A against x = log|c_A| with a Fermi-Dirac function:
   A(x) = A0 / (exp((x−μ)/T) + 1).
   - Warm-start each time step from the previous one's fit.
   - Reject any fit whose μ lands outside [min x − 10, max x + 10]. Such fits have run away and give nonsense.
5. Report μ(t), T(t) and R²(t).
6. **Phase diagnostic μ_ph:** unwrap arg ψ(s,t) along *time* separately for each configuration,
   subtract the mean over s, then apply the same WHT and fit. This needs dt ≤ 0.02 at these energy
   scales. Check by halving dt.
7. **Fit-free cross-checks:**
   - `w_hi`: fraction of Σ c_A² sitting in clusters with |A| > 4.
   - `minlog`: min_s log(|ψ|²·2^N), which flags near-nodes of ψ.
   - Half-cut entanglement S.

**Open choice, A0 (please confirm).** The notebook uses `1024` for L=10, which is 2·N_real.
Its markdown says A0 = 2^{L−1} = N_real.
- With A0 = 2·N_real, μ is pinned to the *smallest* coefficient, so it tracks the tail.
- With A0 = N_real, μ is the midpoint of the rank curve.

The tests compute both (`mu` uses N_real, `mu2` uses 2·N_real). Both show the same qualitative onset. The plan defaults to A0 = N_real.

## 2. What the quick tests showed

Calibration: 1D TFI, L=12, gs at g=1.5 → g_f. My code reproduces the known picture (`cal_1d.png`):
- At g_f = 0.5, μ ≈ −17 stays flat until t ≈ 0.45, then jumps to −6.
- The jump coincides with a near-node of ψ (`minlog` drops to about −20).
- **Mechanism:** an amplitude ψ(s,t) approaches zero, so log|ψ| becomes singular and the
  coefficient spectrum becomes dense. This is the microscopic face of a DQPT.

Findings in 2D (4x3):

| Family | Plateau then jump? | Range of t*(p) | Notes |
|---|---|---|---|
| \|+x⟩ → XXZ (no field) | no | ≈0.1 everywhere | Trivially static at Jxx=0 (pure phase) and at Jxx=J (SU(2) eigenstate) |
| \|+x⟩ → 2D TFI, h_f | weak | 0.07–0.23 | μ rises smoothly from t≈0.05. A product start in 2D gives no clean plateau (`cuts_tfi2d_plus.png`) |
| **A: gs(−ZZ−4X) → −ZZ − h_f X** | **yes** | 0.24–0.57, and **∞ for h_f ≳ 3.3** | Onset only for quenches across the 2D QCP (h_c ≈ 3.04). t* grows toward h_c and toward 0. Cleanest physics |
| **B: gs(−ZZ−4X) → −ZZ + Jxx(XX+YY) − 1·X** | partial | **0.05–0.77** (10× range) | Uses Jxx/J as the control, as requested. Peaks at Jxx ≈ −0.7 near the SU(2) ferromagnet. μ starts moving at t ≈ 0.02–0.05 whenever Jxx ≠ 0, so the plateau is short |
| B with h_f = 2, 3, 4 | similar | 0.05–0.8 plus ∞ windows | Larger h_f opens "never hard" windows around Jxx ∈ [−0.7, 0.1] |
| C: field quench h 4 → 1 at fixed Jxx (gs uses the same Jxx) | no | 0.16–0.34 | Only the SU(2)-point slowdown. For Jxx > 0.5 the *initial* gs already has μ₀ ≈ −5. Discarded |
| Pure XXZ with an Mz-eigenstate start | not usable | — | U(1) gives exact zeros, so log|ψ| = −∞ (an existing caveat). Needs \|+x⟩ or a U(1)-breaking field |

**RBM check (α=1, full-sum TDVP, RK45, 8 runs, about 5–10 min each on 1 core; `rbm_vs_mu.png`).**
Qualitative agreement holds in both families:
- **A:**
  - h_f = 3.65: infidelity stays at 1e-5 and μ stays flat.
  - h_f = 2.96: infidelity saturates at 4e-4 and μ saturates at −10.
  - h_f = 0.89: infidelity reaches 0.3 and μ jumps to −5 at t ≈ 0.2.
  - h_f = 0.2: slow rise in both.
- **B:**
  - Jxx = 0.95: infidelity → 1 by t = 0.3.
  - Jxx = 0.14: infidelity 0.5 at t = 1.
  - Jxx = −0.41 and −0.95: infidelity ≈ 2e-3.
  - At Jxx = −0.95, μ first jumps to −10 and then *plateaus*. The infidelity does the same at 1e-3.

**Caveat to address.** μ correlates strongly with entanglement:
- Spearman(μ, S) = 0.90 for B and 0.82 for A, against 0.65 in 1D.
- Spearman(μ, μ_ph) ≥ 0.93, so the phase proxy adds little so far.

The study has to show that μ predicts infidelity *beyond* S. Family B contains points where the two decouple: at Jxx = −1.5, μ ≈ −5 (hard) while S ≈ 0.9 (low).

## 3. Recommended family

**Primary (B): XXZ ferromagnet in a transverse field, controlled by Jxx/J.**

  H_f = −J Σ⟨ij⟩ Z_i Z_j + Jxx Σ⟨ij⟩ (X_i X_j + Y_i Y_j) − h_f Σ_i X_i,  with J = 1 and h_f = 1,
  ψ₀ = ground state of the 2D TFI paramagnet, −J Σ ZZ − h_i Σ X, with h_i = 4 (finite fluctuations, so μ₀ ≈ −15).

- **Why this family:**
  - It is 2D with volume-law entanglement growth, reaching S ≈ 3.4 out of a maximum of 4.2 at 12 sites, so MPS will not scale.
  - It interpolates between the TFI (Jxx = 0), the SU(2) ferromagnet (Jxx = −J, slow dynamics), and the easy-axis/AFM-XY regime (Jxx > 0, fast scrambling).
  - Transverse-field XXZ models appear in quantum simulators (Rydberg/dipolar) and in anisotropic
    magnets (not sure about a specific material match; check before claiming one).
- **Baseline (A): 2D TFI, control h_f/J (Jxx = 0).** This is the direct 2D analogue of the existing
  1D study and has the cleanest "no onset when not crossing h_c" signature. Run it as the reference cut.
- **Stretch step:** scan the 2D (Jxx, h_f) plane, then pick the one-dimensional cut that maximises
  the range of t* while keeping a plateau (t_dev ≥ 0.1). The h_f = 3–4 scans suggest
  that cuts at larger h_f give longer plateaus and "never hard" windows.

## 4. Execution plan for the agent

Main agent: Opus. It owns every physics or modelling decision and every interpretation.
Subagents get only mechanical work with an exact spec. **At most 3 subagent spawns in total.**
Respect CLAUDE.md: ≤ 30 CPUs and ≤ 40 GB in total. Set `OMP_NUM_THREADS=1` and
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` per process, and run ≤ 28 processes at once.

1. **Consolidate the code** (main agent, small).
   - Promote `quick_tests/qt.py` (builders for H and ψ₀, `evolve`, `analyze`) and the
     NetKet operator builder from `rbm_quick.py` into one module `quench2d.py` next to `ed_definitions.py`.
   - Keep the self-check: `python quench2d.py` asserts FWHT correctness, the SU(2) eigenstate, and that the NetKet and dense H agree.
   - Make A0 a single switch.
2. **ED scans** → *Sonnet subagent #1*. The spec is the exact commands plus output paths under `../data/quench2d/`.
   - Family B: Jxx ∈ linspace(−1.5, 1.5, 31), t ∈ [0, 1.5], dt = 0.01.
   - Family A: h_f ∈ linspace(0.2, 4, 20), same time grid.
   - Optional (Jxx, h_f) plane: 16 × 6 points at dt = 0.02.
   - Save μ, T, R², μ2, μ_ph, w_hi, minlog and S as npz, plus the standard heatmaps.
   - Robustness: rerun 3 points at dt/2 to check μ_ph unwrapping, and at h_i = 6 to check sensitivity to the initial state.
   - The subagent reports file paths and any fit failures. It does not interpret.
3. **Define t*** (main agent).
   - Primary: t*_μ = first t with μ > μ₀ + 5.
   - Secondary: t*_w = first t with w_hi > 1%.
   - Also record t_dev (μ > μ₀ + 1.5) as the plateau length.
   - Choose 8–10 representative values of p that span the range of t*, including the "never hard" points.
4. **RBM+TDVP runs** → *Sonnet subagent #2*. Use `rbm_quick.py`, generalised. Settings:
   - α = 1, complex128, RK45 adaptive with dt = 0.01, QGTJacobianDense(holomorphic=True).
   - Initial state from 1000 SR steps of infidelity minimisation (the existing recipe).
   - Run family B on the full 31-point grid, family A on 10 points, and α ∈ {1, 2} at 5 representative points.
   - Roughly 70 runs at 5–10 min each on 28 cores, so about 30 min wall time.
   - The subagent launches, monitors, retries crashed runs once, and reports a table of final infidelities. It does not interpret.
5. **Comparison figures** → *Haiku subagent #3*. Exact plotting spec only:
   - heatmaps of μ(t, p) and log₁₀ infidelity(t, p) side by side;
   - overlay of t*_μ(p) and t*_inf(p), where t*_inf is the first t with infidelity > 1e-3;
   - scatter of μ against log₁₀ infidelity, coloured by S.
6. **Analysis** (main agent).
   - Spearman correlation of μ with log infidelity over all (t, p).
   - **Partial correlation controlling for S**. This is the key test of whether μ beats entanglement.
   - The same with μ_ph, T and w_hi.
   - Check t*_μ against t*_inf.
   - Check that the 1D calibration still holds with the final code.
   - Write `note_quench2d.md`: the model, the figures, and an honest verdict, including where μ fails (e.g. early-time Jxx ≠ 0 drift).

**Success criteria:**
- t*_μ(p) and t*_inf(p) agree in ordering, with Spearman ≥ 0.8 across p.
- "Never hard" points in μ have infidelity < 1e-3 at t = 1.5.
- μ keeps significant predictive power after controlling for S.

## 5. Known risks

- **Small system:** 12 sites with a 3-site open direction. DQPT-like features are smeared, and
  h_c(4x3) is not the thermodynamic 3.04. Treat t* as a crossover time.
- **TDVP error accumulation:** infidelity mixes expressivity with integration and QGT-regularisation error. Since we decided on TDVP only, keep the diag_shift and solver tolerances fixed across p so the comparisons are fair.
- **Short plateau in B:** μ starts moving at t ≈ 0.02–0.05 whenever Jxx ≠ 0, because the sudden
  change in Jxx acts immediately. If the plateau is essential, fall back to the best cut in the (Jxx, h_f) plane (step 2, optional) or to family A.
