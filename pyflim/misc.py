from functools import wraps

from numpy import asanyarray


def array_args(func):
    """Convert function inputs to numpy arrays."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        arrays = map(asanyarray, args)
        kw_arrays = {k: asanyarray(v) for k, v in kwargs.items()}
        return func(*arrays, **kw_arrays)

    return wrapper


def complex_to_real(x):
    """Convert complex to 2D real vector."""

    return x[..., None].view(float)
