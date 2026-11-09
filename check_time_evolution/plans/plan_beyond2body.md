# Plan: beyond-two-body quench families, ED screen and report (2026-10-30)

Goal: find quench families with multi-spin interactions and a control parameter, where some quenches are predicted
easy for a plain RBM and some hard. Stage 1 (this plan) is ED only: compute the difficulty predictors, plot them
as $(g, t)$ heatmaps for all Hamiltonians, and write a report. Stage 2 (RBM runs) is submitted later by the user
and edits the same report.

Decisions from Q&A
- Plain RBMs only, no extra Jastrow or 3-body visible terms (stage 2).
- Multi-spin terms are a goal in themselves.
- A quench that a known Clifford or local-unitary frame makes classical does not count (so no frame
  interpolation; S2 is excluded). Non-local dualities (Wegner, toric code to TFI) are allowed.
- Starts: product states or ground states.
- ED at $N = 12$–$18$. MPS at larger sizes only later, for families that pass.

Why ground-state starts: from a product start the RBM has $W \approx 10^{-3}$. The $k$-body part of
its TDVP tangent space is then of order $W^{k-1}$, so 3- and 4-body terms freeze it (I5, report l. 1190).
A ground state of a correlated $H_0$ gives $O(1)$ weights. Product starts stay in the screen for comparison,
and the families B6 and B7 add a 2-body term that can build up $W$.

## 1. Hamiltonians (IDs for `rydberg_screen.py`)

Notation: $A_s = \prod_{i \in s} X_i$ (star), $B_p = \prod_{i \in p} Z_i$ (plaquette).
$\triangle$ = all triangles, $\triangle_{\uparrow}$ = up-triangles only, $\square$ = square plaquettes.
$h_0$ = field of the pre-quench ground state, $h_f$ = field after the quench. Energy unit: $J = 1$.

| ID | $H(g)$ | lattice ($N$) | start | control $g$ |
|---|---|---|---|---|
| B1 | toric code $-\sum_s A_s - \sum_p B_p - h_f \sum_i Z_i$ | tc12, tc18 | gs($h_0 = 0.1$) | $h_f$ |
| B2 | toric code $- h_f \sum_i (Z_i + X_i)/\sqrt 2$ | tc12, tc18 | gs($h_0 = 0.1$, same direction) | $h_f$ |
| B3 | Xu–Moore: $-\sum_\square ZZZZ - h \sum_i X_i$ | sq4x4 (16) | $\lvert{+x}\rangle$; gs($h_0 = 1.5$) | $h$ resp. $h_f$ |
| B4 | Baxter–Wu: $-\sum_\triangle ZZZ - h \sum_i X_i$ | tri6x3 (18) | $\lvert{+x}\rangle$; gs($h_0 = 1.5$) | $h$ resp. $h_f$ |
| B5 | Newman–Moore: $-\sum_{\triangle_\uparrow} ZZZ - h \sum_i X_i$ | tri4x4 (16) | $\lvert{+x}\rangle$; gs($h_0 = 1.5$) | $h$ resp. $h_f$ |
| B6 | Ising + plaquette: $-\sum ZZ - K \sum_\square ZZZZ - 0.5 \sum X$ | sq4x4 (16) | $\lvert{+x}\rangle$ | $K$ |
| B7 | Baxter–Wu + Ising: $-\sum_\triangle ZZZ - \lambda \sum ZZ - 0.5 \sum X$ | tri4x4 (16) | $\lvert{+x}\rangle$ | $\lambda$ |

Reference rows (existing families, recomputed so they have the new predictors):
G4 (2D TFI from gs($h = 4$), sq4x3; the TFI benchmark), R1 (Rydberg Ising, sq4x3, $\lvert{+x}\rangle$; the best RBM-easy
case), S1 (toric code $+\epsilon \sum Z$, tc12, tilted product), I5 (Baxter–Wu + field, tri4x3, $\lvert{+x}\rangle$).

Notes on each family
- B1: dual to the 2D TFI with $h_c \approx 1/3.044 \approx 0.33$, so it is G4 in a different basis.
  This tests how much the predictors depend on the basis.
- B2: on the diagonal field direction, no duality is known and the model is non-integrable
  (Fradkin–Shenker diagonal; transition near $h \approx 0.34$–$0.42$, not sure).
- B3: self-dual at $h = 1$, first-order transition (Xu–Moore 2004; Orús et al. 2009, not sure). On the ordered side
  the ground state is about $2^{2L-1}$-fold degenerate (subsystem symmetry), so the gs start is on the paramagnetic side.
- B4: Baxter–Wu needs $L_x, L_y$ multiples of 3 for its 3-sublattice order, hence the new tri6x3; tri4x3 frustrates
  it (not sure how strongly). Self-dual at $h = 1$ (not sure).
- B5: for $L = 2^k$ the classical Newman–Moore ground state is unique (not sure), so tri4x4 is used. Self-dual at $h = 1$.
- B6, B7: from a product start, the 2-body term builds up $W$. $K = 0$ or $\lambda = 0$ reduces to I4 or I5.

Grids (12 points each)
- B1, B2: $h_f \in$ {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.4, 0.5, 0.66, 1.0}; $t \in [0, 10]$.
- B3–B5, $\lvert{+x}\rangle$: $h \in$ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0}; $t_{\max}$ = `tmax_field(h)`.
- B3–B5, gs: $h_f \in$ {0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0}; $t \in [0, 10]$.
- B6: $K \in$ lin(0, 2, 12). B7: $\lambda \in$ lin(0, 2, 12). $t_{\max} = 20$.
- 201 save times as usual. In total about 144 new tasks plus 4 reference families (about 50 tasks at $N = 12$).

## 2. Predictors

Ranked as in the predictor summary (2026-10-30). All are computed from the exact $\psi(t)$.

| Symbol | Name | Status | Figure |
|---|---|---|---|
| $R_2$ | few-body error rate | **new code** | main |
| $\Phi_2 = \int_0^t R_2 \thinspace dt'$ | accumulated few-body error | new, from $R_2$ | main |
| $w_{hi}$ | high-order cluster weight (orders $> 4$) of $\log\lvert\psi\rvert$ | exists (`analyze`) | main |
| $\mu$ | cumulant potential | exists | main |
| $S$ | half-cut entanglement entropy [nats] | exists | main (MPS-hardness) |
| $\mu_{\mathrm{ph}}$, $w_{hi}^{\mathrm{ph}}$ | phase versions | exist | appendix |
| $R_4$, $\Phi_4$ | few-body rate with clusters up to 4 sites | **new code** | appendix |
| $\eta$, $\chi_\epsilon$ | non-classicality, MPS bond dimension | exist | appendix / row labels |

$R_2$ definition (paper draft): with $p(s) = \lvert\psi(s)\rvert^2$ and the local velocity
$v(s) = -i \thinspace (H\psi)(s)/\psi(s)$,
$R_k^2 = \min_c \sum_s p(s) \thinspace \lvert v(s) - f_c(s) \rvert^2$, where $f_c(s) = \sum_{\lvert A \rvert \le k} c_A s_A$
(all clusters up to $k$ sites, all-to-all, complex $c_A$).

Implementation (new `few_body_rate(psi, Hpsi, N, k)` in `rydberg2d.py`, called in `run_task`):
- Weighted normal equations through Walsh–Hadamard transforms, so no $2^N \times n_k$ matrix:
  the Gram matrix is $G_{AB} = \hat p(A \oplus B)$ with $\hat p$ = `fwht(p)`;
  the right side is $b_A$ = `fwht`$(-i \thinspace \bar\psi \thinspace H\psi)$ at $A$ (real and imaginary parts separately).
- Solve $G c = b$ with `lstsq` (rcond $10^{-12}$; $G$ is singular when $p$ has zeros, e.g. near the toric-code ground state).
- Residual evaluated directly, not as $\lVert H\psi \rVert^2 - b^\dagger G^{-1} b$ (that cancels badly when $R$ is small):
  $f_c$ = `fwht`(c padded to $2^N$), then
  $R_k^2 = \sum_s \lvert -i (H\psi)(s) e^{-i\arg\psi(s)} - \lvert\psi(s)\rvert f_c(s) \rvert^2$.
- Sizes: $k = 2$ has 172 unknowns at $N = 18$; $k = 4$ has 4048. The $k = 4$ solve takes an estimated 2–5 s per time.
- Stored as `rate2`, `rate4` (the key `R2` is already the $R^2$ of the $\mu$ fit in `analyze`; not renamed, to keep
  old files readable). $\Phi_k$ is computed in the figure code (cumulative trapezoid).
- Checks (asserts in `rydberg2d.__main__`):
  - $R_2(t) = 0$ to $10^{-10}$ for a diagonal 2-body $H$ (R0, $\Omega = 0$);
  - $R_2(0) = 0$ for G4 (gs start of a 2-body $H$, field quench: $v(0)$ is exactly 2-body);
  - $R_2$ equals a dense `lstsq` on the explicit $2^N \times n_2$ basis at $N = 8$.

Expected behaviour, to read the heatmaps:
- For $H_f = H_D + h_f \sum X$ from gs($H_D + h_0 \sum X$), $v(s, 0) = \mathrm{const} + (1 - h_f/h_0) H_D(s)$.
  So $R_2(0) > 0$ whenever $H_D$ has 3- or 4-body terms (B3–B5 gs, B6, B7), and $R_4(0) = 0$.
  B1 has $R_2(0) = 0$ (the quench term is one-body).
- From $\lvert{+x}\rangle$, $R_2(0) > 0$ for every multi-body $H$. This matches the freeze argument above.
- $R_2$ was validated only on the 1D TFIM, where the RBM represents 2-body phases exactly.
  For multi-body $H$ it measures the distance to a 2-body Jastrow motion, which is stricter than what an RBM with
  $O(1)$ weights can do; $R_4$ is the multi-body-aware variant. Which one tracks the RBM is a question for stage 2.

Thresholds and comparability across rows
- $w_{hi}$: onset marker at $w_{hi} = 1\%$; transfers across $N$ without rescaling (1D and 2D).
- $\mu$: marker at $\mu = -10$; the scale drifts with $N$ ($N = 12$–$18$ here), so rows of different $N$ are not
  comparable in $\mu$. Stated in the captions.
- $R_2$: the paper's $\alpha = 1$ threshold 0.15 at $L = 10$ with its size rescaling. The direction of the rescaling
  ($\sqrt{10/L}$ as quoted in the summary vs $\sqrt{L/10}$ for an extensive $R_2^2$) has to be checked in the paper
  before it is drawn. Until then, no $R_2$ threshold line.
- $S$: MPS-hard marker at $S_{\mathrm{Page}}(N)/2$ (1.83 nats at $N = 12$).

Optional, cheap: validate $R_2$, $\Phi_2$, $w_{hi}$ against the **existing** RBM data (`data/rbm_long.csv`, families G4,
R1, D3, I1 at $N = 12$): Spearman $\rho$ with $\log_{10} I$, and onset times vs $t^\ast_{\mathrm{inf}}$. No new RBM runs.
This is the first 2D test of $R_2$ and tells how far to trust its heatmaps. Appendix.

## 3. Code changes (minimal)

1. `rydberg2d.py`:
   - lattices `tc{Lx}x{Ly}` (generalised from tc12; tc12 = tc2x3, tc18 = tc3x3) and `tri6x3`;
   - helpers `up_triangles(lat)` and `squares(lat)` next to `triangles()`;
   - `few_body_rate()` as above;
   - asserts in `__main__`: toric code $[A_s, B_p] = 0$ and gs energy $-2 L_x L_y$ at $h = 0$; up-triangle count $= N$;
     square count $= N$ on sq4x4; the $R_k$ checks.
2. `rydberg_screen.py`:
   - B1–B7 in `FAMILIES` (tier 4) and `spec()`;
   - `run_task` stores `rate2`, `rate4`, and for gs starts the gap $E_1 - E_0$ of $H_0$ (warning below $10^{-4}$:
     near-degenerate ground state, `eigsh` may return a mixture);
   - `RAW` and the gather output directory from an env variable (default unchanged), so this screen writes to
     `data/b2b/` and the reference families are recomputed there without touching `data/screen/`.
3. `rydberg_figs.py`: generalise `main_rows` to a list of panels and an optional RBM column; add the B labels to
   `HAM` and `MODEL`. Figure code follows the python-plotting skill.

## 4. Runs

- `NPROC=28` (below the 30-CPU limit). Cost (estimate, not measured):
  - $N = 16$: existing sq4x4 runs take minutes per task; $N = 18$ is 4 times larger per step.
  - $R_4$ adds about 201 × (2–5 s) ≈ 7–17 min per task. Total: roughly 2–4 h.
  - Memory: well below 40 GB ($2^{18}$ amplitudes = 4 MB per state; $G$ for $k = 4$ is 260 MB).
- First a smoke test: one $g$ per family at the cheapest lattice, and the asserts. Then the full run.

## 5. Report (with the `report` skill)

New LaTeX report `check_time_evolution/report/beyond2body_report.tex` (shares `refs.bib` with the quench2d report;
figures in `rydberg/figs/b2b/`).

- After the runs: propose 2–3 highlight families and wait for the user's choice (report skill, step 2).
  Likely candidates: the family with the clearest easy/hard split along $g$ at high $S$; the toric code (B1 vs G4,
  same physics in two bases); a negative case.
- Main text (about 6 pages):
  - Abstract and Overview: question, method, answer.
  - Graphical abstract: for each highlight, the fraction of the $(g, t)$ plane that is predicted easy
    and MPS-hard, against $g$.
  - Definitions: $R_2$, $\Phi_2$, $w_{hi}$, $\mu$, $S$, $\eta$, the onset times.
  - Results: one section per highlight, with the heatmap matrix of the highlights only
    (columns $R_2$ | $\Phi_2$ | $w_{hi}$ | $\mu$ | $S$; onset lines of each predictor drawn on every panel, as in
    `main_rows`; model inset with $H$, start and lattice).
  - Verdict: which families have a window that is predicted RBM-easy and MPS-hard, at what $\eta$; limitations.
    The prediction is not yet tested with RBMs; that is stage 2.
- Appendices:
  - the full matrix (all families and reference rows, same columns);
  - a second matrix ($\mu_{\mathrm{ph}}$, $w_{hi}^{\mathrm{ph}}$, $R_4$, $\Phi_4$, $\chi_\epsilon$);
  - Hamiltonians and lattices (this table);
  - the $R_k$ implementation and checks;
  - the optional $R_2$ validation on existing RBM data;
  - numerical details (time steps, gap check, cost); deviations from this plan; a notation table.
- Review pass with a subagent (report skill, step 6); `review.md` next to the report. Compile until clean.

## 6. Stage 2 (later, submitted by the user; not part of this run)

RBM TDVP runs (plain RBM, $\alpha = 1, 2$) on the families the report selects. They add the RBM infidelity column and
$t^\ast_{\mathrm{inf}}$ to the same heatmap figures, and test which predictor ($R_2$, $R_4$, $w_{hi}$, $\mu$) tracks the
RBM on multi-body Hamiltonians. They edit this report (results, verdict) instead of starting a new one.
The report should leave room for this: the heatmap code takes an optional RBM column, and the verdict says explicitly
what is only predicted.

## 7. Risks

- Near-degenerate ground states (ordered or topological phases on small tori): hence the gap check, and gs starts on
  the paramagnetic side except B1/B2 ($h_0 = 0.1$, topological; splitting about $h_0^L \approx 10^{-3}$ at $L = 3$,
  not sure).
- tc12 (2×3) is very small and anisotropic; tc18 is the main lattice.
- $\mu$ depends on the basis and on $N$; $R_2$ is untested outside the 1D TFIM. The report treats the heatmaps as
  predictions, not as results about RBMs.
- The $R_2$ threshold rescaling must be checked in the paper before it is used.
