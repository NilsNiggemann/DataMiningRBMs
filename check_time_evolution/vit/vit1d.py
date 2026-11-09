"""Small 1D ViT neural quantum state (open boundaries, complex128 parameters throughout).

Spins -> patches of size b -> linear embed (d) -> `layers` pre-LN blocks
(factored attention with a learned n x n matrix alpha per head, then GeLU FFN)
-> z = sum over patches -> log psi = sum_a log cosh(b_a + w_a . z), complex w, b.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn


class ViT1D(nn.Module):
    b: int = 2
    d: int = 8
    heads: int = 2
    layers: int = 1
    ffn: int = 2
    nhid: int = 0  # output hidden units; 0 -> d

    @nn.compact
    def __call__(self, s):
        s = jnp.atleast_2d(s).astype(jnp.complex128)
        B, L = s.shape
        assert L % self.b == 0, "L must be divisible by patch size b"
        n, d, h = L // self.b, self.d, self.heads
        assert d % h == 0
        x = nn.Dense(d, param_dtype=jnp.complex128)(s.reshape(B, n, self.b))
        for i in range(self.layers):
            y = nn.LayerNorm(param_dtype=jnp.complex128)(x)
            v = nn.Dense(d, use_bias=False, param_dtype=jnp.complex128)(y).reshape(B, n, h, d // h)
            alpha = self.param(f"alpha{i}", nn.initializers.normal(1 / n), (h, n, n), jnp.complex128)
            a = jnp.einsum("hij,bjhc->bihc", alpha, v).reshape(B, n, d)
            x = x + nn.Dense(d, use_bias=False, param_dtype=jnp.complex128)(a)
            y = nn.LayerNorm(param_dtype=jnp.complex128)(x)
            y = nn.Dense(self.ffn * d, param_dtype=jnp.complex128)(y)
            x = x + nn.Dense(d, param_dtype=jnp.complex128)(nn.gelu(y))
        z = x.sum(axis=1)
        m = self.nhid or d
        u = nn.Dense(m, param_dtype=jnp.complex128)(z.astype(jnp.complex128))
        return jnp.sum(jnp.log(jnp.cosh(u)), axis=-1)


LADDER = {  # name: kwargs (b=2)
    "V0": dict(d=4, heads=1, layers=1, ffn=1),
    "V1": dict(d=8, heads=2, layers=1, ffn=2),
    "V2": dict(d=8, heads=2, layers=1, ffn=4),
    "V3": dict(d=16, heads=2, layers=2, ffn=4),
}


def n_real(model, L):
    v = model.init(jax.random.PRNGKey(0), jnp.ones((1, L)))
    return sum(x.size * (2 if jnp.iscomplexobj(x) else 1) for x in jax.tree_util.tree_leaves(v))


if __name__ == "__main__":
    L = 10
    for name, kw in LADDER.items():
        m = ViT1D(**kw)
        v = m.init(jax.random.PRNGKey(0), jnp.ones((1, L)))
        s = jnp.sign(jax.random.normal(jax.random.PRNGKey(1), (5, L)))
        out = m.apply(v, s)
        assert out.shape == (5,) and jnp.iscomplexobj(out) and bool(jnp.all(jnp.isfinite(out)))
        assert m.apply(v, s[0]).shape == (1,)
        print(name, kw, "real N_p =", n_real(m, L), " (2*2^L =", 2 * 2**L, ")")
    print("ok")
