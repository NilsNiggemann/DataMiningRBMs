# Review log (round 1, independent reviewer)

Verdict: REVISE. 4 major, 11 minor findings; all fixed.

- **M1 single 1e-6 floor inconsistent with alpha=0.5 plateau** — fixed: per-run floor (RBM > 10x its t=0 value); all statistics recomputed (597 points, 93 at g_f=0.5).
- **M2 "any branch" overclaim; mechanism stated as fact; amplitude vs phase** — fixed: "the two branch searches we tried"; mechanism labelled as interpretation; phase part stated as dominant (0.16 vs 0.01); "a few configurations per near-miss".
- **M3 pooled rho inflated by shared growth in t; g_f claim unsupported** — fixed: added rho_t (fixed-time rank correlation across 24 runs, 0.93); verdict narrowed to time/alpha dependence within a quench and ordering of runs at fixed time; offset variation (1.7-27) stated.
- **M4 key tables in appendix** — fixed: Tables 1-2 moved into Sec. 5 with [t].
- m1 phi vs varphi clash; psi_n undefined — model renamed psi_K; psi_n defined in Definitions.
- m2 Sec. 4 opening stated local optima as fact — hedged; support choice named as alternative cause.
- m3 "same continuation as TDVP" — now "analogous ... unlike TDVP, refit to exact psi(t)".
- m4 unsupported "consistent with Ref. [1]" — removed; TDVP explanation now "one possible reason".
- m5 rms around equality mixes in offset — replaced by spread around mean (0.27 / 0.62 decades).
- m6 "does not suffer from near-zeros" — replaced by rho >= 0.98 for g_f <= 0.7.
- m7 g_f=0 not in setup — listed in Setup as self-check, "not shown" in Sec. 3.
- m8 inconsistent time ranges / near-zero numbers — grids stated in App. A and captions; "from t~0.25", "peaking at 516 near t~0.7".
- m9 Fig. 3 colours clash with Fig. 2; appendix title; missing model insets — viridis for alpha, grey for excluded points; appendix retitled; model line added to Figs. 1 and 3.
- m10 internal file name and undefined mu — file name moved to App. C; mu bullet dropped.
- m11 c_emptyset in sum / "for log|psi|" ambiguous — now sum over A != emptyset, "its real part log|psi| alone".

# Review log (round 2, after applying the updated report skill; independent reviewer)

Changes before review: braket/cleveref/tcolorbox; graphical abstract (Fig. 1); definition and finding boxes; technical details to footnotes; Tables 1-2 replaced by Fig. 4 (tables to App. C); notation table (App. A); fixed model colours in text and figures.

Verdict: REVISE. 1 major, 14 minor; 13 fixed, 1 partly fixed (page budget).

- **M1 truncated RBM "1-2 orders worse" (Sec. 6, box, opener)** — fixed: checked data/truncation.csv, the gap is 0.4-3.5 decades, largest for alpha>=2; text, box and opener now say so, footnote gives per-alpha values.
- m1 page budget (~7.3 pages) — partly: Sec. 4 merged into Sec. 5 as a paragraph, per-time numbers to footnotes, Wrapping shortened. Main text still ~7.5 pages (boxes, Fig. 1); reaching 6 needs moving a result (e.g. Sec. 6) to the appendix — left to the user.
- m2 colours — fixed: winding-flip curve grey dashed (Fig. 2); Fig. 5 exact state grey, tracked-fit stars red, caption states that the RBM is coloured by alpha there.
- m3 box median "from t~0.5" — fixed: median over points above floor, excess grows from t~0.5.
- m4 winding searches "change little" — fixed: offset slightly lower, ranking not improved.
- m5 abstract "about 4" — fixed: median factor 3.7 over all quenches.
- m6 abstract misses third finding — fixed: sentence added.
- m7 overclaims — fixed: "works when continued in time, not when done at each time separately"; "could lower".
- m8 log cosh mechanism stated as fact; theta_j undefined — fixed: hedged ("without having checked it"), theta_j = b_j + sum W s defined, notation table updated.
- m9 notation clashes — fixed: chi for model state; delta t for time step; log_n psi for branch log; K = N_var stated in Sec. 6.
- m10 near-zero undefined — fixed: defined in Sec. 1 (with near-miss), added to notation table.
- m11 naming — fixed: "single-time fidelity fit" everywhere incl. figures; "network branch"; "offset Delta" in the definition.
- m12 captions — fixed: Hamiltonian line in Figs. 1-2; Fig. 4(b) "markers as in (a)".
- m13 Fig. 5 ticks every 4 decades — fixed: every 2 decades.
- m14 absolute home path — fixed: file name only.
