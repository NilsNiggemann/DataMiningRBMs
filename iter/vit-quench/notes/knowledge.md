# Knowledge (one line per entry, ends with step number)

## Established
- RBM L=10 alpha=1 (240 real params) TDVP: I(t=1)=7.3e-3 (g_f=0.5), ~7e-4 (g_f=3); I(0)=1.4e-6. (000)
- RK4 dt=0.01 is not converged vs RK45 (g_f=3: 1.8e-3 vs 7.1e-4); dt=0.005 gives 7.6e-4. Use dt=0.005. (000)
- ViT ground-state fit (SR lr 0.05, 3000 it): complex body V0 (290 params) 7.4e-3 fails; V2 (1716) 2.8e-6 passes; V1 (1172) 3.9e-6 at it 1800. (000)
- Complex-body V2 TDVP, RK4 dt=0.005, rtol_smooth 1e-6 (~440 s/g_f, 4 threads): g_f=0.5 I=0.9 at t=1; g_f=1,2,3 NaN at steps 127,142,93; control drifts to 5.8e-4. (000)
- rtol_smooth 1e-4..1e-2 removes control drift (flat 2.8e-6) but g_f=0.5 stays 0.57 at t=0.5, so not just regularisation. (000)

## Dead ends
- Real-body ViT V1 (658 params) in TDVP: NaN in 1-3 steps for g_f=0.5..3, control grows exponentially. (000)
- Adam is disallowed (user rule). (000)

## Gotchas
- LayerNorm is non-holomorphic: QGT/SR need holomorphic=False; unclear whether nkx.TDVP then solves the real-parameter equation (not verified). (000)
- Pipeline check not yet done: RBM through the ViT driver with identical settings. (000)

## Where things are
- Drivers/logs in check_time_evolution/vit/ (ptvmc_*.log, pilot_*.log, figs_ptvmc_infidelity_mu.png); data in data/vit_tdvp/ and data/vit_tdvp_gpu/. (000)
