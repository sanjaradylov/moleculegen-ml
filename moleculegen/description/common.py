"""
Common feature transformers.

Classes
-------
OneHotEncoder
    One-hot encoder functor.
"""

__all__ = (
    'OneHotEncoder',
)


import mxnet as mx


class OneHotEncoder:
    """One-hot encoder class. It is implemented as a functor for more
    convenience, to pass it as a detached embedding layer.

    Parameters
    ----------
    depth : int
        The depth of one-hot encoding.
    """

    def __init__(self, depth: int):
        self.depth = depth

    def __call__(self, indices: mx.np.ndarray, *args, **kwargs) -> mx.np.ndarray:
        """Return one-hot encoded tensor.

        Parameters
        ----------
        indices : nx.np.ndarray or mx.sym.Symbol
            The indices (categories) to encode.
        *args, **kwargs
            Additional arguments for `nd.one_hot`.
        """
        if isinstance(indices, mx.sym.Symbol):
            return mx.sym.one_hot(
                indices.as_nd_ndarray(), self.depth, *args, **kwargs).as_np_ndarray()
        # noinspection PyUnresolvedReferences
        return mx.npx.one_hot(indices, self.depth, *args, **kwargs)
