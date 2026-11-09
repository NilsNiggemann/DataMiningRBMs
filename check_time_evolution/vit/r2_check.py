"""Step 0: r^2(0) of the ViT tangent space for the generator i*C(s) and C(s); frozen-state infidelity."""
import sys, os; sys.path[:0]=['..']; os.environ.setdefault("JAX_PLATFORMS","cpu")
import numpy as np, jax, jax.numpy as jnp, flax, netket as nk
import ed_definitions as ed
from vit1d import ViT1D, LADDER
L=10; rung=sys.argv[1]; model=ViT1D(**dict(LADDER[rung],b=2))
_,hi,Hi=ed.build_TFI_hamiltonian(L,1.5,1.0,False); _,psi0=ed.ground_state(Hi)
vs=nk.vqs.FullSumState(hi,model,seed=1234)
vs.variables=flax.serialization.from_bytes(vs.variables,open(f"../../data/vit_tdvp/init_L{L}_{rung}_s1234.msgpack","rb").read())
S=jnp.asarray(hi.all_states()); P=vs.parameters
leaves,td=jax.tree_util.tree_flatten(P)
def unflat(x):
    out=[];i=0
    for l in leaves:
        n=l.size
        if jnp.iscomplexobj(l): out.append((x[i:i+n]+1j*x[i+n:i+2*n]).reshape(l.shape)); i+=2*n
        else: out.append(x[i:i+n].reshape(l.shape)); i+=n
    return jax.tree_util.tree_unflatten(td,out)
x0=jnp.concatenate([jnp.concatenate([l.real.ravel(),l.imag.ravel()]) if jnp.iscomplexobj(l) else l.ravel() for l in leaves])
f=lambda x: model.apply({**vs.variables,"params":unflat(x)},S)
J=np.asarray(jax.jacfwd(f,holomorphic=False)(x0)) if False else np.asarray(jax.jacfwd(lambda x: jnp.stack([f(x).real,f(x).imag]))(x0))  # (2,N,Np)
lp=np.asarray(f(x0)); p=np.exp(2*lp.real); p/=p.sum(); w=np.sqrt(p)[:,None]
Jc=J[0]+1j*J[1]; Jc=Jc-(p[:,None]*Jc).sum(0)
s=np.asarray(S); C=(s[:,:-1]*s[:,1:]).sum(1); dC=C-(p*C).sum()
for name,tgt in [("i*C",1j*dC),("C",dC+0j)]:
    A=np.concatenate([(w*Jc).real,(w*Jc).imag]); b=np.concatenate([(np.sqrt(p)*tgt).real,(np.sqrt(p)*tgt).imag])
    sv=np.linalg.svd(A,compute_uv=False)
    for rt in [1e-10,1e-8,1e-6,1e-4,1e-3,1e-2]:  # rtol on eigenvalues of S = sigma^2
        a=np.linalg.lstsq(A,b,rcond=np.sqrt(rt))[0]
        print(rung,name,"rtol=%g r2=%.3g rank=%d"%(rt,np.sum((A@a-b)**2)/np.sum(b**2),(sv**2>rt*sv[0]**2).sum()))
# frozen state infidelity
H0=ed.build_TFI_hamiltonian(L,0.5,1.0,False)[2]; ev,V=ed.full_diagonalization(H0)
for t in [0.125,0.375,0.5,1.0]:
    ps=ed.evolve_state(ev,V,psi0,np.array([0,t]))[1]; print("g_f=0.5 t=%g I_frozen=%.4f"%(t,1-abs(np.vdot(psi0,ps))**2))
