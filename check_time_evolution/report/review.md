# Review log: Sec. 10 "R1 in an optimised local basis" (2026-10-28)

## Round 1 (independent reviewer, verdict REVISE, 5 major / 10 minor)

| # | finding | response |
|---|---------|----------|
| 1 | J not invariant under one-body phases: `np.angle` at t=0 wraps the one-body phase of the rotated product state into many-body 2π jumps that the time unwrapping keeps (J at θ=0 varied from −11.85 to −4.4) | **Fixed (code bug).** `basis_opt.proxies` now starts the unwrapping from the exact one-body phase of U(0)\|+x⟩; `check()` asserts that a random pure R_z leaves μ_max unchanged. All optimisations and RBM tests rerun (old results in `rydberg/data/basis/old_branch/`). The new optimum at t=1.65 is −14.13 (was −13.74), and the fixed basis moved to a different basin (θ≈0.05, φ≈−1). The reported conclusions changed: all rotated bases now help at early times. The deviations at φ≈−0.21 and +0.10 seen on the grid were not reproduced at the exact angles; the caption now says that the speckle near θ=0 comes from fits at single grid points. |
| 2 | t*_prox does not order t*_inf across bases; the "drop of 2 → 2–8×" relation was inconsistent | Rewritten as "Proxy across bases": μ_max at fixed early t separates σ^z from the rotated bases, as I does; within the rotated bases, neither μ_max nor t*_prox ranks them. |
| 3 | "integration error" does not follow from supervised < TDVP | Fixed: "error of the TDVP evolution, not missing expressivity; projection and integration error not separated". |
| 4 | "unchanged / within 1.5×" contradicted the numbers | Fixed with the rerun numbers: late-time ratios 0.9–1.6 (TDVP) and 0.8–2.0 (supervised), "within a factor 2"; abstract and verdict say "less than a factor 2". |
| 5 | Abstract sentence missing Ω/V_nn, α and units | Fixed. |
| 6 | "up to 8×" inconsistent | Replaced by per-basis maxima over 0.5≤t≤2.5 (6×, 8×, 11× at t=1.35). |
| 7 | Star description wrong; star colour too close to blue | Caption: "μ (a) and μ_ph (b) of the basis that minimises μ_max at that time"; stars now vermillion with a black edge. |
| 8 | θ range and "rows coincide" claims off | Angles are shown in the canonical form 0≤θ≤π/2, −π<φ≤π, with the equivalences explained in the caption; the text quotes \|θ_i\|; the rows claim was removed. |
| 9 | Notation clashes J (Ising coupling) and W (RBM weights) | Renamed: objective μ_max, rotation U, frame field ν_i (text and figures). |
| 10 | Why max(μ, μ_ph) and not μ; optimised vs. reported times | One sentence of motivation added; "the optimisations use 1–25 times; table and figures validate on all 200". |
| 11 | Supervised t*_inf differences within seed spread | Seed ranges are given; with the rerun, the rotated ranges (2.25–3.0) lie above the σ^z range (1.95–2.1). |
| 12 | t_S missing | Added: t_S = 1.35, RBM window 1.2 → 1.6–1.8. |
| 13 | Plan jargon, runtime sentence | Removed ("This follows plans/…", "Deviations from the plan", "no Optuna", runtime). |
| 14 | "–" undefined; η range | "–: not run" is in the caption; η is quoted as 0.052–0.074 from the rerun. |
| 15 | "σ^z close to optimal" too broad | Now "close to optimal at late times". |

## Round 2 (fresh reviewer, verdict REVISE, 2 major / 6 minor; round-1 responses M2–M5 confirmed)

| # | finding | response |
|---|---------|----------|
| M1 | Fig. 10(c,d) heatmaps scrambled: `pool_map` used `imap_unordered`, and the grid scan reshaped results by completion order; the "speckle" caption sentence (and the round-1 response on it) was wrong | **Fixed (code bug).** `pool_map` now uses the ordered `imap`; the three stored grids were recomputed (minima unchanged, now at the right coordinates) and the figure redrawn. The speckle sentence was removed. The optimisation was not affected, since the grid starts used matching (x, J) pairs. |
| M2 | "all rotated bases lower μ_max by ≈2 at t≲2.5" is false: the fixed basis raises μ_max before t≈1 (outside the optimisation window 1.2≤t≤30) | Fixed: "near t≈1.3–2.1 by 1.2–2.4; before t≈1 the fixed basis raises it by up to 2.3" (recomputed); the optimisation window is stated in "Objective"; Overview: "around t≈1.5". |
| m1 | At t≤0.75, other rotated frames are also worse | Fixed (TDVP 5× λ=0.01, 3× fixed; supervised 2.4× λ=0.01). |
| m2 | Late drop "0.2–0.4" is inaccurate | Now "typically 0.2 (0–0.8)". |
| m3 | 2°–7° does not hold for U(t) | "per-time and fixed: 2°–7° (time-dependent: up to 12°)". |
| m4 | Canonical-form statement covers (c,d) wrongly | The caption separates (a,b) canonical from (c,d) full range. |
| m5 | Imprecise "U(t) within 0.4"; t*_prox for λ=1 | Fixed. |
| m6 | η range includes the untested λ=0.1 | "0.052–0.070 in the tested bases". |
