# Handoff: small Vision-Transformer NQS cannot follow 1D TFI quenches (status 2026-11-03)

Purpose of this document: a self-contained summary for an LLM that will do online literature/implementation research and suggest what to try next. Facts marked **(measured)** come from runs in this project; **(hypothesis)** means not verified.

## 1. Scientific question

- Existing study (RBM): time evolution of the 1D transverse-field Ising (TFI) chain after a quench, using a neural quantum state (NQS) and time-dependent variational principle (TDVP). For some quenches the RBM's infidelity $\mathcal I(t)$ grows large ("hard"), for others it stays small ("easy"), giving a heat map of $\log_{10}\mathcal I$ over $(g_f, t)$.
- Question: do the RBM-hard quenches stay hard for a Vision Transformer (ViT) ansatz? Is there a *small* ViT (number of real parameters $N_p \lesssim 2^L$, far below the $2\cdot 2^L$ real numbers of the full state vector) that follows the quench with low infidelity?
- Plan: start at $L=10$, then go to $L=16$–$20$ with exact (full-sum) evaluation, no Monte Carlo. Constraint: at most 30 CPUs and 40 GB RAM (shared machine).

## 2. Physical setup (all measured / fixed by the existing RBM code)

- Hamiltonian: $H(g) = -J\sum_i Z_i Z_{i+1} - g\sum_i X_i$, $J=1$, **open boundaries** (so no translation invariance).
- Initial state: ground state at $g_i = 1.5$. Quench to $g_f \in [0,3]$ (40-point grid in the RBM study), evolve for $t \le 1$.
- Infidelity: $\mathcal I = 1 - \lvert\langle\psi\vert\phi\rangle\rvert^2$ for normalised vectors, $\psi$ the NQS state, $\phi$ the exact state.
- Exact reference: full diagonalisation for $L\le 12$, `expm_multiply` on the sparse Hamiltonian above that.
- RBM reference run (old study): $L=10$, $\alpha=1$ (240 real parameters), complex parameters, `FullSumState`, adaptive RK45, holomorphic QGT; initial infidelity $1.4\times10^{-6}$.

## 3. Software stack

- NetKet (JAX), `nkx.TDVP`, `FullSumState` (exact sum over all $2^L$ configurations), `QGTJacobianDense`, linear solver `nk.optimizer.solver.pinv_smooth` with `rtol_smooth`.
- Ground-state preparation ("fit"): minimise the infidelity to the exact ground state with stochastic reconfiguration (SR, `nk.optimizer.SR`, `diag_shift=5e-4`) and plain `Sgd` (learning rate `lr`). The `InfidelityOperator` from `netket.experimental.observable` is used with a `LogStateVector` target.
- Float64 / complex128 everywhere. CPU only.
- User rule: **never use Adam** for NQS fits; only vary SR learning rate and diag shift.

## 4. The ViT ansatz as implemented (`check_time_evolution/vit/vit1d.py`)

1. Spins $s\in\lbrace\pm1\rbrace^L$ are cast to complex and split into $n=L/b$ patches of size $b=2$, each embedded linearly (dense layer) into dimension $d$.
2. `layers` pre-LayerNorm encoder blocks: $x \mathrel{+}= W_O\,\mathrm{Attn}(\mathrm{LN}(x))$, then $x \mathrel{+}= \mathrm{Dense}(\mathrm{GeLU}(\mathrm{Dense}(\mathrm{LN}(x))))$ with FFN width $\mathrm{ffn}\cdot d$.
   - Attention is **factored** (no query/key): learned weights $\alpha^{h}_{ij}$ of shape (heads, n, n) multiply values $V=W_V\,\mathrm{LN}(x)$. Initialised normal with scale $1/n$. Full learned matrix because open boundaries break translation invariance.
3. Pooling: $z=\sum_p x_p$ (sum over patches).
4. Output: $\log\psi(s)=\sum_{a=1}^{d}\log\cosh(b_a+w_a\cdot z)$ with complex $w_a, b_a$ (RBM-like head).
5. No symmetrisation. No sign/phase rule imposed (the TFI ground state is positive in the $Z$ basis; real-time evolution generates complex phases).
6. Version history:
   - v1: all body parameters real (float64), only the output head complex.
   - v2 (current): **all parameters complex128** (including LayerNorm scales, attention $\alpha$, dense weights). `n_real` counts a complex parameter as 2 reals.
   - `holomorphic=False` for QGT and SR (LayerNorm uses conjugation and is not holomorphic).

### Parameter ladder at $L=10$ (current complex body, $b=2$, real parameter counts)

| Rung | $d$ | heads | layers | ffn | $N_p$ (real) |
|---|---|---|---|---|---|
| V0 | 4 | 1 | 1 | 1 | 290 |
| V1 | 8 | 2 | 1 | 2 | 1172 |
| V2 | 8 | 2 | 1 | 4 | 1716 |
| V3 | 16 | 2 | 2 | 4 | 11656 |

(The earlier real-body counts were 165 / 658 / 930 / 6100.) Reference: $2^{10}=1024$, $2\cdot 2^{10}=2048$, RBM 240.

## 5. Driver (`check_time_evolution/vit/vit_tdvp_quench.py`)

- Options: `--ansatz {vit,rbm} --L --rung V0..V3 (or --d --heads --layers --b --ffn) --gf ... --tmax --dt --ode {rk4,rk45} --rtol_smooth --fit_only --n_iter --lr --seed --out`.
- Fit cached as msgpack in `data/vit_tdvp/init_L{L}_{arch}_s{seed}.msgpack`; one npz per $(L,\text{arch},g_f,\text{ode},dt)$ with `times`, `infidelity`, `n_params`, etc.
- `--fit_mc` (sampled fit for $L=20$) is not implemented.
- Python environment: `check_time_evolution/.venv/bin/python`.

## 6. Results so far (all measured)

### 6.1 Integrator check (RBM, $L=10$, $g_f\in\lbrace0.5,1.5,3.0\rbrace$)
- $g_f=1.5$ is the control (no quench): $\mathcal I$ stays flat at $\approx 1.8\times10^{-6}$ for all integrators.
- RK4 with $dt=0.01$ disagrees with adaptive RK45: max relative deviation of $\mathcal I(t)$ of 3.4 ($g_f=0.5$) and 83 ($g_f=3$). Final values at $g_f=3$: RK45 $7.1\times10^{-4}$, RK4($dt=0.01$) $1.8\times10^{-3}$, RK4($dt=0.005$) $7.6\times10^{-4}$.
- RK4 with $dt=0.005$ deviates by 0.01 ($g_f=0.5$) and 0.24 ($g_f=3$). Decision: use $dt=0.005$ everywhere. Remaining 24% may come from RK45 or `pinv_smooth` noise (unknown).
- RBM final infidelities at $t=1$: $g_f=0.5$: $7.3\times10^{-3}$; $g_f=3$: $\approx7\times10^{-4}$.

### 6.2 Ground-state fits at $L=10$ (SR, no Adam)
- First attempt ($lr=0.01$, 1000 iterations, real-body ViT): V0 0.038, V1 $3.4\times10^{-5}$, V2 0.020, V3 0.99. The loss starts near 1 and escapes slowly, i.e. slow rather than stuck.
- With $lr=0.05$, 3000 iterations, real body: V0 $7.3\times10^{-3}$, V1 $1.2\times10^{-6}$ ($N_p=658$), V2 reached $4.8\times10^{-5}$ at iteration 1500 (timed out), V3 0.89 (stuck).
- Complex body (same settings): V0 $7.4\times10^{-3}$ ($N_p=290$, fails), V1 $3.9\times10^{-6}$ at iteration 1800 (timed out, not cached), V2 $2.8\times10^{-6}$ ($N_p=1716$, passes), V3 not refitted.
- So V0 (comparable size to the RBM) cannot even represent the initial ground state to $10^{-5}$, but 1172–1716 parameters can.

### 6.3 Real-body ViT (V1, $N_p=658$) in TDVP: failed
- With `rtol_smooth` $10^{-8}$ and $10^{-6}$, RK4 $dt=0.005$: NaN within 1–3 steps for $g_f=0.5,1,2,3$; $g_f=0$ jumps to $\mathcal I\approx 1$; control $g_f=1.5$ grows exponentially from $1.2\times10^{-6}$ to $3.7\times10^{-5}$ over $t\le1$ (RBM control stays flat).
- Control for short time $t\le0.2$ was stable for `rtol_smooth` $10^{-6},10^{-4},10^{-3}$.
- Interpretation (hypothesis): real parameters in the body mean the tangent space reaches complex phases only through the final complex layer; stiff, ill-conditioned dynamics with explicit RK4.

### 6.4 Complex-body ViT V2 ($N_p=1716$, $\mathcal I(0)=2.8\times10^{-6}$), $t\le1$, RK4 $dt=0.005$, `rtol_smooth` $10^{-6}$
About 440 s per $g_f$ with 4 threads.

| $g_f$ | behaviour |
|---|---|
| 0 | $\mathcal I$ rises to $\approx0.07$ at $t=0.125$ and to 1 by $t\approx0.5$ |
| 0.5 | rises to 0.03 (t=0.125), 0.26 (0.375), 0.9 (t=1) |
| 1.0 | 0.008 (t=0.125), 0.14 (t≈0.625), then NaN at step 127 |
| 1.5 (control) | slow drift: $3.8\times10^{-6}$ (t=0.125) up to $5.8\times10^{-4}$ (t=1) |
| 2.0 | 0.013 (t=0.125), 0.99 (t≈0.375), NaN at step 142 |
| 3.0 | 0.75 at t=0.125, 0.998 later, NaN at step 93 |

### 6.5 `rtol_smooth` scan for V2 ($t\le0.5$)
| `rtol_smooth` | control $g_f=1.5$, $\mathcal I(0.5)$ | $g_f=0.5$, $\mathcal I(0.5)$ |
|---|---|---|
| $10^{-4}$ | $2.9\times10^{-6}$ (flat) | 0.57 |
| $10^{-3}$ | $2.8\times10^{-6}$ (flat) | 0.57 |
| $10^{-2}$ | $2.8\times10^{-6}$ (flat) | 0.57 |

Larger smoothing removes the control drift but leaves the quench error unchanged (0.57), so the failure on the quench does not look like regularisation noise. (Hypothesis: genuine limitation of the tangent space / architecture.)

## 7. Open questions and untested ideas

1. **Pipeline check (cheapest):** run the RBM through exactly the same driver and settings ($dt=0.005$, `rtol_smooth` $10^{-4}$) at $g_f=0.5$. If it gives about $10^{-2}$ at $t=1$, the ViT failure is genuine. Not yet done.
2. Is the failure specific to this design? Candidates to change:
   - The head: $\sum_a\log\cosh$ on a pooled vector $z$ of dimension $d$ (only $d=8$ hidden units for V2) is much narrower than the RBM's $\alpha L = 10$... but the RBM has $\alpha L$ hidden units acting directly on the spins, while here all information passes through the $d$-dimensional pooled embedding. Try `nhid` larger than $d$.
   - Sum-pooling versus a per-patch or per-site read-out (e.g. a complex log-amplitude per patch summed, as used in published ViT wave functions).
   - Full QKV attention rather than the factored $\alpha_{ij}$.
   - No explicit sign/phase structure.
3. TDVP solver choices: complex-valued ansatz with `holomorphic=False` yields a real-parameter (McLachlan) TDVP that may differ from the holomorphic TDVP. Check whether NetKet's `TDVP` solves the real-parameter or complex-parameter equation for mixed/non-holomorphic ansätze, and whether the imaginary part of the QGT is being dropped.
4. Initial-state quality: $\mathcal I(0)\approx3\times10^{-6}$ may still leave a force component outside the tangent space; check the TDVP residual and the QGT spectrum (condition number) of V2 versus the RBM.
5. Integrator: explicit RK4 on a stiff projected ODE; consider implicit midpoint / projected-evolution (p-tVMC) or adaptive step with tight tolerances.
6. Scaling plan (not started): $L=16$, $18$, $(20)$ with exact sums; cost measured at $L=20$: one RBM TDVP right-hand side about 19 s and 17 GB, ViT-sized proxy about 20 s and 30 GB (16 cores, loaded machine). RK4 with $dt=0.005$ at $L=20$ would need about 4.4 h per $g_f$.

## 8. Suggested online research (please search and report concrete, citable answers)

- Published use of **transformer / ViT NQS for real-time dynamics** (as opposed to ground states): which architecture details, parameter counts, and TDVP/solver settings did they use? Are they complex-valued throughout? Which quenches were reported to work?
- Known **failure modes of TDVP with transformer or LayerNorm-based ansätze**: ill-conditioned QGT, non-holomorphic layers, regularisation (diag shift / `pinv` cutoff), projected time-dependent VMC (p-tVMC, Sinibaldi et al.), and how they handle small networks.
- Whether published ViT wave functions use **sum-pooling to a $d$-dimensional vector followed by a $\log\cosh$ head**, or another read-out; compare to the Viteritti–Rende–Becca ViT wave function and Rende et al. work on large-scale transformer NQS optimisation.
- Evidence on **expressivity vs. parameter count for dynamics**: is the entanglement/"phase" generation in real-time evolution known to need larger networks than ground-state representation? (RBM cannot do all quenches either; the original question is about the RBM-hard region.)
- NetKet-specific: handling of **complex parameters with non-holomorphic layers** in `nkx.TDVP` and `QGTJacobianDense` (real vs. complex mode), and known pitfalls.
- Alternatives to compare against at small cost: MPS/TEBD baselines at $L=10$–$20$ to cross-check what "hard" means.

## 9. Files

- Plan and log: `check_time_evolution/plans/plan_vit_quench.md` (checklist S0–S16; S0–S3 done, S4 partly done, S5 failed, S6 onward on hold).
- Code: `check_time_evolution/vit/vit1d.py`, `check_time_evolution/vit/vit_tdvp_quench.py`.
- Data: `data/vit_tdvp/` (caches and npz); integrator-check data in the session scratchpad.
- RBM reference: `check_time_evolution/rbm_tdvp_quench.py`, `data/cumulants_Fermi_dirac_TDVP/rbm_tdvp_quench.npz`; heat-map script `check_time_evolution/quench2d/tfi1d_heatmaps_mumax.py`.
