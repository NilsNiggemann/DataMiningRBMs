# Next: finish 002-ptvmc-heatmap-vs-mumax (jobs running since 2026-11-07 14:55 UTC)
Pending: 28 local procs (14 RBM 30 it, 14 ViT 100 it), out data/vit_tdvp_scan/{rbm,vit}; ViT low-g_f expected ~5-6 h.
Check: `ps -eo args | grep -c vit_ptvmc_quench`; each log vit/logs/scan_*.log ends with t=1.000.
Tasks: 1) when done run quench2d/tfi1d_heatmaps_ptvmc.py, view preview (600 px). 2) Compare I vs mu_max (which mu_max level where I crosses 1e-4/1e-2; RBM vs ViT).
3) summary, knowledge, report (+figs), publish, commit.
