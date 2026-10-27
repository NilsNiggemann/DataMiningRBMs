import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qt import *
from concurrent.futures import ProcessPoolExecutor
TM = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
times = np.arange(0, TM + 1e-9, TM / 200)
# name: (geom, vals, H_f(p) kwargs, psi0 spec)  geom=(Lx,Ly,pbcx,pbcy); psi0 spec = 'plus' or kwargs for gs
FAM = {
 'tfi1d_plus': ((12,1,False,False), np.linspace(0.1,2,10), lambda p: dict(J=-1,hx=-p), 'plus'),
 'tfi2d_gs4':  ((4,3,True,False), np.linspace(0.2,4,12), lambda p: dict(J=-1,hx=-p), dict(J=-1,hx=-4.0)),
 'tfi2d_gs6':  ((4,3,True,False), np.linspace(0.2,4,12), lambda p: dict(J=-1,hx=-p), dict(J=-1,hx=-6.0)),
 'xxz2d_gs4':  ((4,3,True,False), np.linspace(-1.5,1.5,12), lambda p: dict(J=-1,Jxx=p,hx=-1.0), dict(J=-1,hx=-4.0)),
 'tfi2d_plus':  ((4,3,True,False), np.linspace(0.25,5,12), lambda p: dict(J=-1,hx=-p), 'plus'),
 'xxzh1_plus':  ((4,3,True,False), np.linspace(-1.5,1.5,12), lambda p: dict(J=-1,Jxx=p,hx=-1.0), 'plus'),
 'xxzh05_plus': ((4,3,True,False), np.linspace(-1.5,1.5,12), lambda p: dict(J=-1,Jxx=p,hx=-0.5), 'plus'),
 'xxzh1_gs6':   ((4,3,True,False), np.linspace(-1.5,1.5,12), lambda p: dict(J=-1,Jxx=p,hx=-1.0), dict(J=-1,hx=-6.0)),
}
for h in [2.0, 3.0, 4.0]:
    FAM[f'xxzh{h:g}_gs4'] = ((4,3,True,False), np.linspace(-1.5,1.5,12), (lambda h: lambda p: dict(J=-1,Jxx=p,hx=-h))(h), dict(J=-1,hx=-4.0))
for h in [1.0, 2.0]:
    FAM[f'fq{h:g}'] = ((4,3,True,False), np.linspace(-1.5,1.0,12), (lambda h: lambda p: dict(J=-1,Jxx=p,hx=-h))(h), 'gs_same')
def run(args):
    fam, p = args
    (Lx,Ly,px,py), vals, kw, init = FAM[fam]
    N=Lx*Ly; nn,nn2 = lattice(Lx,Ly,px,py)
    if init == 'gs_same': init = dict(J=-1, Jxx=p, hx=-4.0)
    psi0 = plus_state(N) if init=='plus' else ground_state(hamiltonian(N,nn,nn2,**init))
    return analyze(evolve(hamiltonian(N,nn,nn2,**kw(p)), psi0, times), N, times)
if __name__ == '__main__':
    fam = sys.argv[1]; vals = FAM[fam][1]
    with ProcessPoolExecutor(len(vals)) as ex:
        res = list(ex.map(run, [(fam,p) for p in vals]))
    grid = {k: np.array([r[k] for r in res]) for k in res[0]}
    np.savez(f'scan_{fam}.npz', vals=vals, times=times, **grid)
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, k in zip(axs, ['mu','mu_ph','minlog','S']):
        im = ax.pcolormesh(times, vals, np.where(np.isfinite(grid[k]), grid[k], np.nan), shading='auto'); fig.colorbar(im, ax=ax); ax.set_title(f'{fam}: {k}')
    plt.tight_layout(); plt.savefig(f'heat_{fam}.png', dpi=65)
    fig, axs = plt.subplots(1, 5, figsize=(22, 3.6))
    for i in range(0, len(vals), 2):
        for ax, k in zip(axs, ['mu','mu_ph','w_hi','minlog','S']):
            ax.plot(times, grid[k][i], label=f'{vals[i]:.2f}'); ax.set_title(f'{fam}: {k}')
    axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f'cuts_{fam}.png', dpi=65)
