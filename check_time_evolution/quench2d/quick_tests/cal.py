import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qt import *
N=12; nn=[(i,i+1) for i in range(N-1)]
times=np.arange(0,1.5+1e-9,0.01)
Hi=-hamiltonian(N,nn,J=1,hx=1.5); psi0=ground_state(Hi)
fig,axs=plt.subplots(1,4,figsize=(18,3.5))
for gf in [0.2,0.5,0.8,1.2]:
    r=analyze(evolve(-hamiltonian(N,nn,J=1,hx=gf),psi0,times),N,times)
    for ax,k in zip(axs,['mu','w_hi','minlog','S']): ax.plot(times,r[k],label=f'gf={gf}'); ax.set_title('1D TFI L12 gi=1.5: '+k)
axs[0].legend(); plt.tight_layout(); plt.savefig('cal_1d.png',dpi=70)

d={f:np.load(f'scan_{f}.npz') for f in ['xxz','tfi']}
fig,axs=plt.subplots(2,3,figsize=(16,7))
for row,f in enumerate(d):
    D=d[f]
    for i in range(0,len(D['vals']),2):
        for ax,k in zip(axs[row],['mu','w_hi','minlog']):
            ax.plot(D['times'],D[k][i],label=f"{D['vals'][i]:.2f}"); ax.set_title(f+': '+k)
    axs[row,0].legend(fontsize=7)
plt.tight_layout(); plt.savefig('cuts.png',dpi=70)
