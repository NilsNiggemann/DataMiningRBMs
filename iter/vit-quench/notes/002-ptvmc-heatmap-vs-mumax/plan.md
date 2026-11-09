# 002 ptvmc-heatmap-vs-mumax (step 001 skipped by user)
Goal: heatmap of p-tVMC infidelity I(g_f,t) for RBM and ViT V2 vs mu_max(g_f,t) (running max of stored cumulant-fit mu_grid).
Tasks: 1) 14 g_f = every 3rd of the 40-pt TDVP grid, g_i=1.5, L=10, dt .025, t<=1, lr .5.
2) RBM 30 it ds1e-4, 14 single-thread procs. 3) ViT V2 100 it ds1e-4, 14 single-thread procs (28 CPUs total, <30).
4) Figure: heatmaps RBM/ViT/mu_max(+I contours) + scatter I vs mu_max. 5) notes, report, publish.
Done when: both heatmaps + comparison with mu_max exist.
