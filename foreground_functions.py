import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo
from powerbox import LogNormalPowerBox, PowerBox
from powerbox.tools import angular_average_nd
from powerbox.dft import fft, fftfreq
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from instrumental_functions import sigma
from scipy import signal
import os

def build_sky(sky_size, n_cells, frequencies , S_min=1e-6, S_max=1e-3, alpha=4100., beta=1.59, gamma=0.8, f0=150e6):
    """
    Create a grid of flux densities corresponding to a sample of point-sources drawn from a power-law source count
    model.
    Notes
    -----
    The sources are populated uniformly on an (l,m) grid. *This is not physical*. In reality, sources are more
    likely to be uniform an angular space, not (l,m) space. There are two reasons we do this: first, it corresponds
    to a simple analytic derivation of the statistics of such a sky, and (ii) it doesn't make too much of a difference
    as long as the beam (if there is one) is reasonably small.
    """
    print("Populating point sources... ")

    # Find the mean number of sources
    n_bar = quad(lambda x: alpha * x ** (-beta), S_min, S_max)[
                0] * sky_size ** 2  # Need to multiply by sky size in steradian

    # Generate the number of sources following poisson distribution
    n_sources = np.random.poisson(n_bar)

    if not n_sources:
        print("There are no point-sources in the sky!")

    # Generate the point sources in unit of Jy and position using uniform distribution
    S_0 = ((S_max ** (1 - beta) - S_min ** (1 - beta)) * np.random.uniform(size=n_sources) + S_min ** (
            1 - beta)) ** (1 / (1 - beta))

    pos = np.rint(np.random.uniform(0, n_cells ** 2 - 1, size=n_sources)).astype(int)

    # Grid the fluxes at reference frequency, f0
    sky = np.bincount(pos, weights=S_0, minlength=n_cells ** 2)
    
    # Find the fluxes at different frequencies based on spectral index
    sky = np.outer(sky, (frequencies / f0) ** (-gamma)).reshape((n_cells, n_cells, len(frequencies)))
    
    # Divide by cell area to get in Jy/sr (proportional to K)
    sky /= ( sky_size / n_cells)**2

    return sky