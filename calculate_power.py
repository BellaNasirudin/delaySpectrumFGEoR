import numpy as np
from astropy import constants as const
from astropy import units as un
from numpy.fft import fftshift, fft, fftfreq, fft2
from scipy.signal import blackmanharris
from unitConversion import mK_to_JyperSr, redshifts_to_frequencies, frequencies_to_redshifts, k_perpendicular, k_parallel, hz_to_mpc, sr_to_mpc2, Gz
from astropy.cosmology import Planck15 as cosmo
from scipy import integrate

def delay_transform(data, frequencies):
	delayTaper_ft = fftshift(fft(data * blackmanharris(np.shape(data)[-1]), axis=-1), axes=-1) * np.diff(frequencies)[0]
	
	eta = fftshift(fftfreq(len(frequencies), np.diff(frequencies)[0]))

	return delayTaper_ft, eta

def calculate_2DdelayPS(data, bl_rotated, frequencies, min_rbin= 25, max_rbin=850, numberBins = 61):

	u = np.outer(bl_rotated[:,0], (frequencies/const.c).value)
	v = np.outer(bl_rotated[:,1], (frequencies/const.c).value)

	delayTaper_ft, eta = delay_transform(data, frequencies)

	r = np.sqrt(u**2 + v**2)
	
	rbins = np.linspace(min_rbin, max_rbin, numberBins)

	sq_delay = np.abs(delayTaper_ft)**2

	psTaper = np.zeros((len(frequencies), len(rbins)-1))
	weights = np.zeros((len(frequencies), len(rbins)-1))

	for ff in range(len(frequencies)):

		weights[ff, :] = np.histogram(r[:,ff], rbins)[0]
		
		psTaper[ff, :] = np.histogram(r[:,ff], rbins, weights = sq_delay[:, ff])[0]


	psTaper[weights>0] /= (weights[weights>0]**2)

	return psTaper, eta, rbins, weights, delayTaper_ft

def obsUnits_to_cosmoUnits(power, sky_size, frequencies, kperp, kpar):
	z_mid = np.mean(frequencies_to_redshifts(frequencies))

	power_cosmo = power * (un.W / un.m**2 / un.Hz ) **2 * un.Hz**2 / mK_to_JyperSr(z_mid)**2 * (hz_to_mpc(frequencies[0], frequencies[-1]))**2 * (sr_to_mpc2(z_mid))**2

	# divide by volume
	power_cosmo = power_cosmo / (sky_size**2 * un.sr * sr_to_mpc2(z_mid) * 10e6 *un.Hz / Gz(z_mid))

	kperp = k_perpendicular(kperp, z_mid).value
	kpar = k_parallel(kpar, z_mid).value

	return power_cosmo, kperp, kpar

def kparallel_wedge(k_perp, field_of_view, redshift):

	E_z = cosmo.efunc(redshift)
	H_0 = (cosmo.H0).to(un.m/(un.Mpc * un.s))

	functionDc_z = lambda x: 1 / cosmo.efunc(x)
	Dc_z = integrate.quad(functionDc_z, 0, np.max(redshift))[0]

	return k_perp * np.sin(field_of_view) * E_z * Dc_z / (1 + redshift) #* H_0 / const.c