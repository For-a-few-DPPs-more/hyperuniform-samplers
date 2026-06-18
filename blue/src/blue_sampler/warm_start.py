import numpy as np

def sobol_init(N: int, D: int, seed: int | None = None) -> np.ndarray:
    """Sobol sequence as an init"""
    try: from scipy.stats import qmc
    except ImportError:
        raise ImportError("Install scipy first to use Sobol sampling")
    engine = qmc.Sobol(d=D, seed = seed)
    sample = engine.random(N)
    return sample