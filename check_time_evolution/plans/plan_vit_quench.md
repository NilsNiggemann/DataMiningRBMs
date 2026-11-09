# Plan: ViT neural quantum states for 1D TFI quenches (2026-11-02)

After approval, this file is copied to `check_time_evolution/plans/plan_vit_quench.md`, which is step S0.

## User prompt (summary)

- Implement a Vision Transformer (ViT) ansatz.
- Test whether 1D TFI quenches that are hard for the RBM stay hard for the ViT, starting with about 10 sites and measuring the infidelity.
- Check whether a *small* ViT that does this exists at all.
- Produce an infidelity heat map like the RBM one.
- Then go to larger L while keeping the numerical cost low.

## Decisions from Q&A

- **Setup:** match the existing RBM heat map. Make it cheaper unless the full grid runs overnight.
- **"Small ViT":** judged with a parameter ladder. If no ViT with $N_p$ well below $2\cdot 2^L$ works at $L=10$, move to $L=14$–$16$.
- **Large L:** local only, $L\le 20$, no cluster. Limits are 30 CPUs and 40 GB.

## Setting (from the existing RBM study)

**Reference figure:** `draftv1.tex:159`, `fig:tfi1d-heat`.
- Data: `check_time_evolution/rbm_tdvp_quench.py`, saved to `data/cumulants_Fermi_dirac_TDVP/rbm_tdvp_quench.npz`.
- Plot: `check_time_evolution/quench2d/tfi1d_heatmaps_mumax.py`.

**Model:** $H(g)=-J\sum_i Z_iZ_{i+1} - g\sum_i X_i$, built by `ed.build_TFI_hamiltonian` (`ed_definitions.py:7`).
- $J=1$, open boundaries.
- The quench starts from the ground state at $g_i=1.5$.

**Grid:** $g_f\in[0,3]$ with 40 values, $t\le 1$, output every $0.01$.

**RBM run:** $L=10$, $\alpha=1$ (about 240 real parameters), `FullSumState`, adaptive RK45. It reaches $\mathcal I(0)=1.4\times10^{-6}$.

**Infidelity:** $\mathcal I = 1 - \lvert\langle\psi\vert\phi\rangle\rvert^2$ on normalised vectors, computed by `Netket/analysis.py:330`.

**Deviations from the user's notes (premise check)**
- Open boundaries break translation invariance. Attention therefore uses a full learned matrix $\alpha_{ij}$ per head, of size $n\times n$.
- No symmetrisation is used.
- The notes' setup (PBC, $T=5$) is not used.

## Feasibility check for $L=20$ (measured 2026-11-02, 16 cores, machine load about 75)

| Quantity | Measured |
|---|---|
| Exact reference: sparse H, `eigsh`, `expm_multiply` over 51 times | 4 s + 6 s + 14 s, 4.5 GB |
| One TDVP right-hand side, RBM $\alpha=1$ (440 complex parameters) | about 19 s, 17 GB |
| One TDVP right-hand side, ViT-sized proxy (`RBMModPhase`, 840 real parameters) | about 20 s, **30 GB** |
| Steps for $t\le1$ at $L=10$, adaptive RK45 (as in the RBM run) | 456 steps, about 2700 right-hand sides |
| Steps for $t\le1$ at $L=10$, RK4 with $dt=0.01$ | 100 steps = 400 right-hand sides |

**Consequences**
- **Adaptive RK45 at $L=20$:** about 15 h per value of $g_f$. It would time out, so it is excluded.
- **RK4 at $L=20$:** about 2.2 h per value of $g_f$, with one process at a time.
- **Fit cost:** a full-sum ground-state fit with 1000 iterations would take about 5.5 h at $L=20$, so the fit there must be cheap.
- **$L=18$:** about 4× cheaper and about 8 GB (estimate). It is the main "large L".
- **Run time:** RK4 with $dt=0.01$ is used everywhere and is validated against RK45 and $dt=0.005$.

## Code (new files in `check_time_evolution/vit/`)

1. **`vit1d.py`, the flax model.**
   - **Embedding:** patches of size $b$, embedded linearly into dimension $d$.
   - **Encoder:** `num_layers` pre-LayerNorm blocks, each made of factored attention followed by a GeLU feed-forward layer of width `ffn`·$d$, with skip connections.
     - Factored attention uses learned weights $\alpha_{ij}$ of shape (heads, n, n), with values from $W_V$ and output from $W_O$.
   - **Output:** $z=\sum_p y_p$ and $\log\psi=\sum_a\log\cosh(b_a+w_a\cdot z)$ with complex $w_a$ and $b_a$.
   - **Precision:** float64.
   - **`__main__` self-check:** asserts on the parameter count, finite output and output shape.
2. **`vit_tdvp_quench.py`, the driver.** It is a parameterised version of `rbm_tdvp_quench.py`.
   - **Options:**
     - `--ansatz {vit,rbm}` `--L` `--d --heads --layers --b --ffn`
     - `--gf` (values or grid indices)
     - `--tmax 1 --dt 0.01 --ode {rk4,rk45}` `--rtol_smooth 1e-8`
     - `--fit_only` `--fit_mc`
   - **Fit:** the infidelity-SR loop from `rbm_tdvp_quench.py:52-69`, with `holomorphic=False` for the ViT. The fitted $\theta_0$ is cached as msgpack, using the pattern in `rydberg/rydberg_rbm.py:113-122`.
     - `--fit_mc` fits with sampled infidelity SR (`MCState`) first, then runs at most 100 full-sum polishing iterations.
   - **TDVP:** `nkx.TDVP`, `QGTJacobianDense(holomorphic=False)`, `pinv_smooth`.
   - **Exact reference:**
     - For $L\le12$: `ed.full_diagonalization` and `ed.evolve_state` (`ed_definitions.py:86,122`).
     - Otherwise: `expm_multiply`.
   - **Output:** one npz per $(L,\text{arch},g_f)$ in `data/vit_tdvp/`, with the RBM file's keys plus `n_params`.
     - It is rewritten at every output time, so a killed job keeps partial data.
3. **`vit_figs.py`, the figures.** It reuses `quench2d/plotstyle.py` and the `pcolormesh`/contour code from `tfi1d_heatmaps_mumax.py:21-28`.
   - **Fig. A:** $\mathcal I(0)$ and the mean $\mathcal I$ against $N_p$ at $L=10$, with a line at $2\cdot2^L$.
   - **Fig. B:** heat maps of $\log_{10}\mathcal I$ for the RBM and the ViT, plus their difference, at $L=10$.
   - **Fig. C:** the per-site $-\ln(1-\mathcal I)/L$ at $t=1$ against $g_f$, for the RBM and the ViT, at $L=10,16,18,(20)$.
4. **`run_batch.sh`:** a copy of `quench2d/data/rbm/run_batch.sh` (`xargs -P`, one retry), with `JAX_PLATFORMS=cpu` and the thread count set per job.

## ViT parameter ladder ($b=2$; counts are estimates, the real ones come from `nk.jax.tree_size`)

| Rung | $d$ | heads | layers | ffn | real $N_p$ (est.) |
|---|---|---|---|---|---|
| V0 | 4 | 1 | 1 | 1 | 165 (RBM: about 240) |
| V1 | 8 | 2 | 1 | 2 | 658 |
| V2 | 8 | 2 | 1 | 4 | 930 |
| V3 | 16 | 2 | 2 | 4 | 6100 (more than $2\cdot 2^{10}$) |

**"Small" means** $N_p\lesssim 2^L$; **"good fit" means** $\mathcal I(0)\le 10^{-5}$.

---

## Checklist: sequential steps

Each step is one submission. Do not start a step before the previous step's **Gate** passes. Wall times are estimates (the machine is shared, expect up to 2× variation).

### Part 0: setup
- [x] **S0. Plan file.** Copy this plan to `check_time_evolution/plans/plan_vit_quench.md` and add a line to `Todo.md`.
- [x] **S1. Model.** Write `vit/vit1d.py` and run `python vit1d.py`.
  - **Gate:** the asserts pass, and the parameter counts for V0–V3 are printed and entered into the ladder table above.
- [x] **S2. Driver plus check against the RBM.** Write `vit/vit_tdvp_quench.py` and run it with `--ansatz rbm --L 10 --ode rk45` on $g_f\in\lbrace 0.5, 1.5, 3.0\rbrace$.
  - **Gate:** $\mathcal I(t)$ matches the matching rows of `rbm_tdvp_quench.npz` to within fit noise (a factor of about 2), and the control $g_f=1.5$ stays flat. About 5 min.
- [x] **S3. Integrator check.** (2026-11-02: **gate failed** for $dt=0.01$: max relative deviation of $\mathcal I(t)$ from RK45 is 3.4 at $g_f=0.5$ and 83 at $g_f=3$; $dt=0.005$ gives 0.01 and 0.24. Control $g_f=1.5$ flat. Decision: use $dt=0.005$ everywhere, double the cost estimates. The remaining 24% at $g_f=3$ may be RK45 or `pinv_smooth` noise (not sure).) Same as S2 with `--ode rk4 --dt 0.01` and with `--dt 0.005`.
  - **Gate:** RK4 at $dt=0.01$ agrees with RK45 to within 10% in $\mathcal I(t)$. If it does not, switch to $dt=0.005$ for all later steps and double the cost estimates.

### Part 1: $L=10$, is there a small ViT?
- [x] **S4. Ground-state fits.** (V2 passes; see Log.) Run `--fit_only` for V0–V3 at $L=10$ and record $\mathcal I(0)$ and $N_p$. About 10 min.
  - **Gate:** at least V2 or V3 reaches $\mathcal I(0)\le10^{-5}$. If not, fix the fit (more iterations, SR lr and diag_shift (no Adam)) before continuing.
- [x] **S5. Pilot quenches.** (Gate branch: **no small ViT at $L=10$**, V2 used further; but the V2 dynamics fail the S7 criteria, see Log.) For every rung that fits, run $g_f\in\lbrace 0, 0.5, 1.0, 1.5, 2.0, 3.0\rbrace$ with $t\le1$ and RK4. Make Fig. A. About 15 min.
  - **Gate (decides whether $L=10$ is informative):**
    - **Small and good:** if some rung with $N_p\lesssim 2^{10}$ fits well, choose it as the "small ViT".
    - **Only large rungs fit:** otherwise record **"no small ViT at $L=10$"**, use V2 for S6, and base the hard/easy verdict on Part 2.
- [ ] **S6. Timing.** Time one full quench with the chosen ViT at $L=10$.
  - **Decision:** if 40 values × the time per run ÷ 28 processes is under 8 h, use the full 40-value grid, otherwise 20 values.
- [ ] **S7. Heat map at $L=10$.** Run the whole grid for the chosen ViT with `run_batch.sh -P 28`, single-threaded jobs. Make Fig. B.
  - **Gate:** no NaN rows, and the control row is flat.
- [ ] **S8. Robustness at $L=10$.** Use 3 values of $g_f$, one each in the hard, intermediate and easy regions of the RBM map. Rerun with `rtol_smooth` $10^{-6}$ and $10^{-10}$, with $dt=0.005$, and with a second fit seed.
  - **Gate:** the hard/easy labels do not change. Any label that does change is reported as an artefact.

### Part 2: larger L (local)
- [ ] **S9. $L=16$ fits and validation.** Fit the chosen ViT (and V2 if it differs) and the RBM. Run the S3 integrator check at $L=16$ for one value of $g_f$. About 1 h.
  - **Gate:** $\mathcal I(0)\le10^{-5}$ for both, and RK4 agrees with RK45.
- [ ] **S10. $L=16$ grid.** Run the ViT and the RBM on 20 values of $g_f$ with $t\le1$, as 7 processes × 4 threads (about 2 GB each). About 3–5 h.
  - **Gate:** the data is complete. Choose 8 values of $g_f$ (spanning the hard and easy regions) for $L=18$.
- [ ] **S11. $L=18$.** Fit, then run the ViT and the RBM on the 8 values of $g_f$, as 4 processes × 7 threads (about 8 GB each). Overnight.
  - **Gate:** the data is complete, with $\mathcal I(0)\le10^{-5}$.
- [ ] **S12. $L=20$ timing gate.** Time one right-hand side with the real ViT at $L=20$ and record the time and peak RSS.
  - **Gate:** at most 30 s and at most 35 GB. Otherwise skip S13 and say so in the report.
- [ ] **S13. $L=20$ (optional).** Fit with `--fit_mc` (sampled fit, then at most 100 full-sum polishing iterations, about 1 h). Then run the ViT and the RBM on 3 values of $g_f$, **one process at a time, each under `timeout 4h`**. About 15 h in total, over two nights.
  - **Gate:** at least the ViT runs are complete. Partial runs are reported up to the time they reached.

### Part 3: wrap-up
- [ ] **S14. Figures.** Make the final Figs. A–C with `vit_figs.py`, following the python-plotting conventions.
- [ ] **S15. Report.** Write `check_time_evolution/report/vit_report.tex` with the `report` skill and do the review pass in `review.md`. It answers:
  1. Is there a small ViT at $L=10$?
  2. Do the RBM-hard quenches stay hard for the ViT at $L=10$, $16$, $18$ and $(20)$?
- [ ] **S16. Commit** on a new branch, only when asked.

## Skipped (add if needed)
- The TDVP residual $r^2$ and the QGT condition number: now added for the initial force only (`vit/r2_check.py`).
- Free fermions, MCState dynamics and the Leonardo cluster. None are needed for $L\le20$ with exact sums.
- PBC with translation-invariant attention.

## Risks
- **Ground-state fit:** a ViT fitted by infidelity SR may plateau above $10^{-5}$. Tune SR lr and diag_shift (no Adam). At $L=20$ the sampled fit may be noisy, in which case $\mathcal I(0)$ becomes the floor.
- **Like-for-like comparison:** the RBM map at $L=10$ used adaptive RK45 and a holomorphic QGT. From S9 on, the RBM is rerun with the same driver and the same RK4 integrator, so the comparison is like for like.
- **$L=20$ cost:** measured with a proxy. The real ViT forward pass may be slower (not sure), hence S12.

## Log: S4/S5 outcome so far (2026-11-03)
- **Real-parameter ViT body:** TDVP unstable (NaN within 1–3 RK4 steps, even for the control). Switched to a complex-parameter body (`vit1d.py`); $N_p$ doubled: V0 = 290, V1 = 1172, V2 = 1716, V3 = 11656.
- **Fits at $L=10$** ($lr=0.05$, 3000 iterations, no Adam): V0 $\mathcal I(0)=7\times10^{-3}$ (fails), V2 $2.8\times10^{-6}$ (passes), V1 reached $3.9\times10^{-6}$ at iteration 1800 but timed out (no cache), V3 not refitted.
- **Pilot, V2, RK4 $dt=0.005$, `rtol_smooth` $10^{-6}$:** the control drifts to $5.8\times10^{-4}$, and $g_f=0,0.5,1,2,3$ lose the state ($\mathcal I\approx0.1$–$1$; three reach NaN).
- **`rtol_smooth` scan** ($10^{-4}$ to $10^{-2}$, $t\le0.5$): the control becomes flat, but $g_f=0.5$ gives $\mathcal I(0.5)\approx0.57$ regardless of the value. This equals the frozen-state value $0.572$ (state that does not move).
- **Diagnosis (`vit/r2_check.py`, measured):** the initial force of every quench is the same function, $i\,C(s)$ with $C=\sum_i s_is_{i+1}$, because $\psi_0$ is an eigenstate of $H(g_i)$. The fraction $r^2(0)$ of it that the pseudo-inverse-truncated tangent space of V2 cannot reproduce is: $0.007$ ($\texttt{rtol}=10^{-10}$), $0.18$ ($10^{-8}$), $0.55$ ($10^{-6}$), $0.998$ ($\ge10^{-4}$); with no cutoff it is $3\times10^{-7}$. The force lies in small-eigenvalue directions of the QGT: a cutoff large enough to stabilise the control freezes the dynamics, a small one is unstable. So the failure is a conditioning trade-off, not (yet shown to be) a lack of representability.
- **Not yet verified (from an external research note):** a complex nearest-neighbour Jastrow factor on top of $\psi_0$ is claimed to follow $g_f=0$ exactly and all $g_f$ to $\mathcal I(0.5)\lesssim4\times10^{-5}$ with 140 parameters.
- **Pending decision:** S6 onward is on hold until the conditioning problem is addressed (options: ViT plus Jastrow term, real body with amplitude/phase read-out, adaptive Heun with QGT error norm, p-tVMC).

## Log: p-tVMC (2026-11-04)
- **Why:** Gravina–Savona–Vicentini (Quantum 9, 1803) use a ViT with the same architecture (real body, complex $\log\cosh$ head, $d=72$, 6 layers) via **p-tVMC** because plain tVMC stiffness grows with the nonlinearity; Sinibaldi et al. (arXiv:2606.05293) use plain tVMC with a transformer of the same size class ($D=72$, 2 layers).
- **Driver:** `vit/vit_ptvmc_quench.py` (full sums, exact step $e^{-iH_f dt}\psi_\theta$, $dt=0.025$, SR infidelity minimisation, 30 inner iterations, backtracking: restore best, halve lr, double diag_shift when the loss rises). Note: `InfidelityOperator` matches the **conjugate** of the target state; the target must be conjugated (invisible for real $\psi_0$).
- **Result, V2 ($N_p=1716$), $t\le1$, measured:** stable and complete, but not accurate. $\mathcal I(t=1)$: 0.11 ($g_f=0.5$), 0.017 ($g_f=1$), 0.079 ($g_f=3$) (RBM TDVP: $7\times10^{-3}$, —, $7\times10^{-4}$). $\mathcal I$ stays $<10^{-4}$ until $t\approx0.6$ for $g_f=0.5, 1$; for $g_f=3$ it rises early ($10^{-3}$ at $t=0.25$, $10^{-2}$ at $t\approx0.4$). The per-step optimisation infidelity grows from $10^{-7}$ to $10^{-3}$ at later times, i.e. the 30-iteration budget is not enough there (budget-limited vs capacity-limited not yet separated).
- Cost: about 6 min per step with 8 threads (about 4 h per $g_f$).
