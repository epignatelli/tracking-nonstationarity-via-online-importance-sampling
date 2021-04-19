import jax
import jax.numpy as jnp


def g_lambda(rewards, values, gamma, lamda):
    def body_fun(carry, x):
        r, v = x
        g = carry
        g = r + gamma * ((1 - lamda) * v + lamda) * g
        return g, g

    rewards = jnp.flip(rewards)
    values = jnp.flip(values)
    g = jax.lax.scan(body_fun, (values[0], (rewards, values)))
    return jnp.flip(g)
