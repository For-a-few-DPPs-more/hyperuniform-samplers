"""
installation:
%%bash
pip install "blue_sampler[keops]" && \
curl -sL https://codeload.github.com/For-a-few-DPPs-more/hyperuniform-samplers/tar.gz/main \
  | tar xz --strip-components=2 --wildcards '*/src/sampler'
"""

from .qmc_sampler import sample_l1, sample_wd

def sample_l2(N, D, **kwargs):
    from blue_sampler.run.run_nufft_keops import _nufft_pipeline_keops
    kwargs.setdefault("verbose", 0)
    return _nufft_pipeline_keops(N, D, **kwargs)

def plot(*args, **kwargs):
    from blue_sampler import plot as _plot
    return _plot(*args, **kwargs)

def plot_structure_factor(*args, **kwargs):
    from blue_sampler import plot_structure_factor as _psf
    return _psf(*args, **kwargs)

__all__ = ["sample_l1", "sample_l2", "sample_wd", "plot", "plot_structure_factor"]
__version__ = "1.0"