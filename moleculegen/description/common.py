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


from mxnet import np, npx


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

    def __call__(self, indices: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return one-hot encoded tensor.

        Parameters
        ----------
        indices : np.ndarray
            The indices (categories) to encode.
        *args, **kwargs
            Additional arguments for `nd.one_hot`.
        """
        # noinspection PyUnresolvedReferences
        return npx.one_hot(indices, self.depth, *args, **kwargs)
