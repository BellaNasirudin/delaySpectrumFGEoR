import numpy as np
from astropy import constants as const
from astropy import units as un
from numpy.fft import fftshift, fft, fftfreq, fft2
from scipy import signal
from unitConversion import mK_to_JyperSr, redshifts_to_frequencies, frequencies_to_redshifts, k_perpendicular, k_parallel, hz_to_mpc, sr_to_mpc2, Gz
from astropy.cosmology import Planck15 as cosmo
from scipy import integrate

def delay_transform(data, frequencies, frequency_taper = signal.blackmanharris):
	delayTaper_ft = fftshift(fft(data * frequency_taper(np.shape(data)[-1]), axis=-1), axes=-1) * np.diff(frequencies)[0]
	
	eta = fftshift(fftfreq(len(frequencies), np.diff(frequencies)[0]))

	return delayTaper_ft, eta

def calculate_2DdelayPS(data, baselines_rotated, frequencies, min_rbin= 25, max_rbin=850, numberBins = 61, frequency_taper = signal.blackmanharris):

	u = np.outer(baselines_rotated[:,0], (frequencies/const.c).value)
	v = np.outer(baselines_rotated[:,1], (frequencies/const.c).value)

	delayTaper_ft, eta = delay_transform(data, frequencies, frequency_taper)

	r = np.sqrt(u**2 + v**2)
	
	rbins = np.linspace(min_rbin, max_rbin, numberBins)

	sq_delay = np.abs(delayTaper_ft)**2

	psTaper = np.zeros((len(frequencies), len(rbins)-1))
	weights = np.zeros((len(frequencies), len(rbins)-1))

	for ff in range(len(frequencies)):

		weights[ff, :] = np.histogram(r[:,ff], rbins)[0]
		
		psTaper[ff, :] = np.histogram(r[:,ff], rbins, weights = sq_delay[:, ff])[0]


	psTaper[weights>0] /= (weights[weights>0]) #**2)

	return psTaper, eta, rbins, weights, delayTaper_ft

def calculate_1DdelayPS(data, baselines_rotated, frequencies, sky_size, min_kbin= 2e-1, max_kbin=1.5, numberBins = 11, frequency_taper = signal.blackmanharris, fg_buffer=10):

	u = np.outer(baselines_rotated[:,0], (frequencies/const.c).value)
	v = np.outer(baselines_rotated[:,1], (frequencies/const.c).value)

	delayTaper_ft, eta = delay_transform(data, frequencies, frequency_taper)

	r = np.sqrt(u**2 + v**2)
	
	kbins = np.linspace(min_kbin, max_kbin, numberBins)

	sq_delay = np.abs(delayTaper_ft)**2

	# need to change everything to cosmo unit
	sq_cosmo, kperp, kpar = obsUnits_to_cosmoUnits(sq_delay, sky_size, frequencies, r, eta)

	k = np.sqrt(kperp**2 + kpar**2)

	# avoid the area where there are foregrounds based on the buffer
	fg_edge = kpar[int(len(frequencies)/2) + fg_buffer]

	# and also the wedge
	# remember that sky size is the diameter in lm so need to find radius and multiply by pi to get to radian 
	fg_wedge = kparallel_wedge(kperp, sky_size / 2 * np.pi, np.min(frequencies_to_redshifts(frequencies)))

	print(k.shape, kpar.shape)

	weights = np.histogram(k[((np.abs(kpar)>=fg_edge) & (np.abs(kpar)>=fg_wedge))], kbins)[0]
		
	power = np.histogram(k[((np.abs(kpar)>=fg_edge) & (np.abs(kpar)>=fg_wedge))], kbins, weights = sq_cosmo[((np.abs(kpar)>=fg_edge) & (np.abs(kpar)>=fg_wedge))])[0]

	power[weights>0] /= (weights[weights>0]) #**2)

	return power, kbins[1:]+np.diff(kbins)[0]/2 , weights

def obsUnits_to_cosmoUnits(power, sky_size, frequencies, kperp, kpar):
	z_mid = np.mean(frequencies_to_redshifts(frequencies))

	power_cosmo = power * (un.W / un.m**2 / un.Hz ) **2 * un.Hz**2 / mK_to_JyperSr(z_mid)**2 * (hz_to_mpc(frequencies[0], frequencies[-1]))**2 * (sr_to_mpc2(z_mid))**2

	# divide by volume
	power_cosmo = power_cosmo / (sky_size**2 * un.sr * sr_to_mpc2(z_mid) * 10e6 *un.Hz / Gz(z_mid))

	kperp = k_perpendicular(kperp, z_mid).value
	kpar = k_parallel(kpar, z_mid).value

	return power_cosmo, kperp, kpar # mK^2 Mpc^3 h^-3, h Mpc^-1, h Mpc^-1

def kparallel_wedge(k_perp, field_of_view, redshift):

	E_z = cosmo.efunc(redshift)
	H_0 = (cosmo.H0).to(un.m/(un.Mpc * un.s))

	functionDc_z = lambda x: 1 / cosmo.efunc(x)
	Dc_z = integrate.quad(functionDc_z, 0, np.max(redshift))[0]

	return k_perp * np.sin(field_of_view) * E_z * Dc_z / (1 + redshift) #* H_0 / const.c