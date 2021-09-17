import numpy as np
from astropy import constants as const
from astropy import units as un
from instrumental_functions import add_instrument_rotation, get_baselines, interpolate_frequencies, get_baselines_rotation
from foreground_functions import build_sky
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import time
from astropy.io import fits
import os
from numpy.fft import fftshift, fft, fftfreq, fft2

from scipy.signal import blackmanharris
import multiprocessing as mp
from unitConversion import mK_to_JyperSr, redshifts_to_frequencies, frequencies_to_redshifts, k_perpendicular, k_parallel, hz_to_mpc, sr_to_mpc2, Gz
from astropy.cosmology import Planck15 as cosmo

start = time.time()

def loading_data(N_components):

	if N_components==0:
		if extension_num == 0:

			beam_perturbed = np.load("../data_OSKAR/beam_allFreqTime0_all_" + file_extension + ".npz")["samples_all"][-N_realizations:] + beam_default
		else:
			beam_perturbed = np.load("../data_OSKAR/beam_allFreqTime0_all_" + file_extension + ".npz")["samples_all"] + beam_default
		print(beam_perturbed.shape)
		return beam_perturbed
	else:
		data = np.load("../data_OSKAR/approximation_KPCA_"+ file_extension + "_Ncomponents%i_Nsample%i_allFreqTime0tanhpoly.npz" %(N_components, N_realizations))

		approximation = data["approximation"].T.reshape((N_realizations, 3, n_cells, n_cells))

		return approximation + beam_default

def calculate_delayPS(data, bl_rotated):

	u = np.outer(bl_rotated[:,0], (frequencies/const.c).value)
	v = np.outer(bl_rotated[:,1], (frequencies/const.c).value)

	delayTaper_ft = fftshift(fft(data * blackmanharris(np.shape(data)[-1]), axis=-1), axes=-1) * np.diff(frequencies)[0]

	r = np.sqrt(u**2 + v**2)
	
	rbins = np.linspace(25, 875, 61)

	sq_delay = np.abs(delayTaper_ft)**2

	psTaper = np.zeros((len(frequencies), len(rbins)-1))
	weights = np.zeros((len(frequencies), len(rbins)-1))

	for ff in range(len(frequencies)):

		weights[ff, :] = np.histogram(r[:,ff], rbins)[0]
		
		psTaper[ff, :] = np.histogram(r[:,ff], rbins, weights = sq_delay[:, ff])[0]


	psTaper[weights>0] /= (weights[weights>0]) #**2)

	return psTaper, eta, rbins, weights, delayTaper_ft

def runEverything(num, beam, sky, frequencies, baselines_SKA, N_timestep, sky_size, tot_daily_obs_time,
		int_time, N_components):

	beam_freq = interpolate_frequencies(np.moveaxis(beam, 0, -1), np.array([150,170,190])*1e6, frequencies)
	
	visibilities, baselines_rotated = add_instrument_rotation(sky, frequencies, baselines_SKA, N_timestep, beam_freq, sky_size, tot_daily_obs_time,
		int_time, effective_collecting_area=300, Tsys=0, integration_time=1000, nparallel=1)

	power, kpar, kperp, weights, delayTaper_ft = calculate_delayPS(visibilities, baselines_rotated)

	np.savez("data/power_Ncomp%i_timesteps%i_realization%i_default_FGEoR-defaultParams-wBeam" %(N_components, N_timestep, num), power=power, kperp= kperp, kpar=kpar)

	return visibilities, baselines_rotated, power, kpar, kperp

def obsUnits_to_cosmoUnits(power, sky_size, frequencies, kperp, kpar):
	z_mid = np.mean(frequencies_to_redshifts(frequencies))

	power_cosmo = power * (un.W / un.m**2 / un.Hz ) **2 * un.Hz**2 / mK_to_JyperSr(z_mid)**2 * (hz_to_mpc(frequencies[0], frequencies[-1]))**2 * (sr_to_mpc2(z_mid))**2

	# divide by volume
	power_cosmo = power_cosmo / (sky_size**2 * un.sr * sr_to_mpc2(z_mid) * 10e6 *un.Hz / Gz(z_mid))

	kperp = k_perpendicular(kperp, z_mid).value
	kpar = k_parallel(kpar, z_mid).value

	return power_cosmo, kperp, kpar

n_cells = 256
sky_size = 20 / 180 #in lm
nu_min = 160
nu_max = 170

frequencies = np.linspace(nu_min, nu_max, 100) * 1e6 #in MHz
tot_daily_obs_time = 0.25
int_time = 15 * 60
N_timestep = int(tot_daily_obs_time * 60 *60 / int_time)

# fps foreground for all frequencies
if os.path.exists("data/box_nu%i_%iMHz.npz" %(len(frequencies), nu_min)):
	box = np.load("data/box_nu%i_%iMHz.npz" %(len(frequencies), nu_min))["box"]
else:
	box = build_sky(sky_size, n_cells, frequencies)
	np.savez("data/box_nu%i_%iMHz" %(len(frequencies), nu_min), box=box)

data = np.load("../SKAdataChallenge/data/lightcone_100MHz_2.0degrees-defaultParams.npz")
lightcone = data["lightcone"]
z = data["lightcone_redshifts"]
boxsize_Mpch = data["cell_size"] * np.shape(lightcone)[0] 

fig, ax = plt.subplots(1,2, figsize=(12,8), sharey=True, gridspec_kw={"hspace":0.11, "wspace":0.03, "right":0.86, "left":0.1, "top":0.91, "bottom": 0.1})

im = ax[0].imshow(lightcone[:,:, 263], aspect='auto', extent=(-boxsize_Mpch / 2, boxsize_Mpch / 2, -boxsize_Mpch / 2, boxsize_Mpch / 2), vmin=0, vmax=100)
im = ax[1].imshow(lightcone[0,:, :], aspect='auto', extent=(np.min(z), np.max(z), -boxsize_Mpch / 2, boxsize_Mpch / 2), vmin=0, vmax=80)

ax[0].set_title(r"$z=%.2f$" %z[263], fontsize=20)
ax[0].set_ylabel(r"x [Mpc]", fontsize=20)
ax[0].set_xlabel(r"y [Mpc]", fontsize=20)
ax[1].set_xlabel(r"$z$", fontsize=20)
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)

cax = fig.add_axes([0.88, 0.1, 0.05, 0.81])
cax.tick_params(labelsize=15)
plt.colorbar(im, cax=cax).set_label(r"$\delta T_B$ [mK]", size=20)
fig.savefig("figures/eor_sky-lightcone")

ionized = np.zeros(len(z))

# check when is lightcone 50% ionized
for num in range(len(z)):
	ionized[num] = np.count_nonzero(lightcone[:,:, num].flatten())

fig = plt.figure()
plt.plot(z, ionized / np.shape(lightcone)[0]**2 * 100, "k")
plt.xlabel(r"$z$", fontsize=20)
plt.ylabel("Reionization Completion [%]", fontsize=20)
plt.hlines(50, z.min(), z.max(), colors= "r", linestyles= "--")

fig.savefig("figures/ReionizationCompletionPercentage")

# raise SystemExit
# read the EoR lightcone data and change to Jy
data = np.load("../SKAdataChallenge/data/lightconeTiled_%iMHz_20degrees-defaultParams.npz" %nu_min) # lightcone_%iMHz_0.5degrees.npz"
lightcone_frequencies = data["frequencies"] #["lightcone_redshifts"]#

lightcone_eor = data["lightcone"] * mK_to_JyperSr(frequencies_to_redshifts(lightcone_frequencies))

# flip so frequency is increasing and interpolate
lightcone_eor = interpolate_frequencies(np.flip(lightcone_eor, axis=-1), np.flip(lightcone_frequencies), frequencies)#, new_ncells = n_cells)

# add eor signal to fg sky
sky = box + lightcone_eor

# load baseline
xy_SKA = np.genfromtxt("../OSKAR-2.7.6/apps/test/data/ska_telescope.tm/layout.txt", delimiter=",")
baselines_SKA = get_baselines(xy_SKA[:,0], xy_SKA[:,1])

eta = fftshift(fftfreq(len(frequencies), np.diff(frequencies)[0]))

beam_default = fits.getdata("../OSKAR-2.7.6/data_SKA/beam_pattern_regularStation_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_I_I.fits", ext=0, memmap=False)[0]

N_realizations = 3000

N_components = 20

print("Prepping stuff took ", (time.time() - start)/60**2, " hours.")

ii = 0

visibilities, baselines_rotated, power, kpar, kperp = runEverything(ii, beam_default, sky, frequencies, baselines_SKA, N_timestep, sky_size, tot_daily_obs_time, int_time, N_components)

# plot power in observational unit
fig, ax = plt.subplots(1, 1, figsize=(9,8), gridspec_kw={"hspace":0.15, "wspace":0.1, "right":0.82, "left":0.07, "top":0.91, "bottom": 0.1})

img = ax.imshow((power[51:]), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e10))
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"$r$", fontsize=22)
ax.set_ylabel(r"$\eta$ [1/Hz]", fontsize=22)

cax = fig.add_axes([0.86, 0.09, 0.03, 0.82])
cb = plt.colorbar(img, cax=cax).set_label(r"P [Jy$^2$ Hz$^{2}]$", size=20)
cax.tick_params(labelsize=15)

fig.savefig("figures/power_FGEoR-defaultParams-wBeam")

# convert power to cosmo unit
power_cosmo, kperp, kpar = obsUnits_to_cosmoUnits(power, sky_size, frequencies, kperp, kpar)

fig, ax = plt.subplots(1, 1, figsize=(9,8), gridspec_kw={"hspace":0.15, "wspace":0.1, "right":0.82, "left":0.07, "top":0.91, "bottom": 0.1})

img = ax.imshow((power_cosmo[51:].value), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e10))
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"$k_\perp$ [$h$ Mpc$^{-1}$]", fontsize=22)
ax.set_ylabel(r"$k_\parallel$ [$h$ Mpc$^{-1}$]", fontsize=22)

cax = fig.add_axes([0.86, 0.09, 0.03, 0.82])
cb = plt.colorbar(img, cax=cax).set_label(r"$P$ [mK$^2$ Mpc$^{3}$ $h^{-3}]$", size=20)
cax.tick_params(labelsize=15)

fig.savefig("figures/power_FGEoR-defaultParamscosmo-wBeam")

fig, ax = plt.subplots(1, 1, figsize=(9,8), gridspec_kw={"hspace":0.15, "wspace":0.1, "right":0.82, "left":0.07, "top":0.91, "bottom": 0.1}) 
img = ax.imshow((power_cosmo[51:].value), origin="lower", extent=(kperp[0]* cosmo.h, kperp[-1]* cosmo.h, kpar[51]* cosmo.h, kpar[-1] * cosmo.h), norm=colors.LogNorm(vmin=1, vmax=1e10)) 
ax.set_xscale('log') 
ax.set_yscale('log') 

ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]", fontsize=22) 
ax.set_ylabel(r"$k_\parallel$ [Mpc$^{-1}$]", fontsize=22) 
 
cax = fig.add_axes([0.86, 0.09, 0.03, 0.82]) 
cb = plt.colorbar(img, cax=cax).set_label(r"$P$ [mK$^2$ Mpc$^{3}$ $h^{-3}]$", size=20) 
cax.tick_params(labelsize=15) 
 
fig.savefig("figures/power_FGEoR-defaultParamscosmo-wBeam-noLittleh") 