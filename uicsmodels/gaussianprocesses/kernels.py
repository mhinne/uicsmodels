import jax
import jaxkern as jk
import jax.numpy as jnp
import jax.random as jrnd
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional
from jaxtyping import Array, Float


class Discontinuous(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0: Float = 0.0) -> None:
        self.base_kernel = base_kernel
        self.x0 = x0
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        def check_side(x_, y_):
            return 1.0*jnp.logical_or(jnp.logical_and(jnp.less(x_, self.x0), 
                                                      jnp.less(y_, self.x0)), 
                                      jnp.logical_and(jnp.greater_equal(x_, self.x0), 
                                                      jnp.greater_equal(y_, self.x0)))

        #
        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())

    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #

#