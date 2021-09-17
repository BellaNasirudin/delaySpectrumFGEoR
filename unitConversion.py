import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo

def mK_to_JyperSr(z):#, cellsize, distances):
    """
    Conversion factor to convert a pixel of mK to Jy/sr (and vice versa via division)
    Taken from http://w.astro.berkeley.edu/~wright/school_2012.pdf
    Parameters
    ----------
    nu : float array, optional
        The mean readshift of the observation.
    
    Returns
    -------
    conversion_factor : float or array
        The conversion factor(s) (per frequency) which convert temperature in Kelvin to flux density in Jy.
    """

    nu = redshifts_to_frequencies(z)

    wvlngth = const.c / (nu / un.s)

    intensity = 2 * const.k_B * 1e-3 * un.K / wvlngth ** 2

    flux_density = 1e26 * intensity.to(un.W / (un.Hz * un.m ** 2))
    
    return flux_density / (1 * un.sr) #* (( cellsize ) / distances)**2

def redshifts_to_frequencies(z):
    """The cosmological redshift (of signal) associated with each frequency"""
    return 1420e6 / (z + 1)

def frequencies_to_redshifts(frequency):
    """The cosmological redshift (of signal) associated with each frequency"""
    return 1420e6 / frequency - 1

def k_perpendicular(r, z):
    '''
    The conversion factor to find the perpendicular scale in Mpc given the angular scales and redshift
    
    Parameters
    ----------
    r : float or array-like
        The radius in u,v Fourier space

    z : float or array-like
        The redshifts
        
    Returns
    -------
    k_perpendicular : float or array-like
        The scale in h Mpc^1
    '''
    k_perpendicular = 2 * np.pi * r / cosmo.comoving_transverse_distance([z]) / cosmo.h
    return k_perpendicular ## [h Mpc^1]

def Gz(z):
    f_21 = 1420e6 * un.Hz
    E_z = cosmo.efunc(z)
    H_0 = (cosmo.H0).to(un.m/(un.Mpc * un.s))
    return H_0 / cosmo.h * f_21 * E_z / (const.c * (1 + z )**2)

def k_parallel(eta, z):
    '''
    The conversion factor to find the parallel scale in Mpc given the frequency scale in Hz^-1 and redshift
    
    Parameters
    ----------
    eta : float or array-like
        The frequency scale in Hz^-1

    z : float or array-like
        The redshifts
        
    Returns
    -------
    k_perpendicular : float or array-like
        The scale in h Mpc^1
    '''

    k_parallel = 2 * np.pi * Gz(z) * eta / (1 * un.Hz)
    return k_parallel ## [h Hz Mpc^-1]

def hz_to_mpc(nu_min, nu_max):
    """
    Convert a frequency range in Hz to a distance range in Mpc.
    """
    z_max = frequencies_to_redshifts(nu_min)
    z_min = frequencies_to_redshifts(nu_max)

    return 1 / (Gz(z_max) - Gz(z_min))

def sr_to_mpc2(z_mid):
    """
    Conversion factor from steradian to Mpc^2 at a given redshift.
    Parameters
    ----------
    z_mid: mean readshift of observation

    Returns
    -------
    """
    return cosmo.comoving_transverse_distance(z_mid)**2 / (1 * un.sr)

def volume(z_mid, nu_min, nu_max, A_eff=20):
    """
    Calculate the effective volume of an observation in Mpc**3, when co-ordinates are provided in Hz.
    Parameters
    ----------
    z_mid : float
        Mid-point redshift of the observation.
    nu_min, nu_max : float
        Min/Max frequency of observation, in Hz.
    A_eff : float
        Effective area of the telescope.
    Returns
    -------
    vol : float
        The volume.
    Notes
    -----
    How is this actually calculated? What assumptions are made?
    """
    # TODO: fix the notes in the docs above.

    diff_nu = nu_max - nu_min

    G_z = (cosmo.H0).to(un.m / (un.Mpc * un.s)) * 1420e6 * un.Hz * cosmo.efunc(z_mid) / (const.c * (1 + z_mid) ** 2)

    Vol = const.c ** 2 / (sigma * un.m ** 2 * nu_max * (1 / un.s) ** 2) * diff_nu * (
                1 / un.s) * cosmo.comoving_transverse_distance([z_mid]) ** 2 / (G_z)

    return Vol.value