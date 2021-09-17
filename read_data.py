import multiprocessing as mp 
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from instrumental_functions import interpolate_frequencies
from astropy import constants as const
from astropy import units as un

def mK_to_Jy(z, cellsize, distances):
    """
    Conversion factor to convert a pixel of mK to Jy/sr.
    Taken from http://w.astro.berkeley.edu/~wright/school_2012.pdf
    Parameters
    ----------
    nu : float array, optional
        The frequencies of the observation (in Hz).
    
    Returns
    -------
    conversion_factor : float or array
        The conversion factor(s) (per frequency) which convert temperature in Kelvin to flux density in Jy.
    """

    nu = redshifts2frequencies(z)

    wvlngth = const.c / (nu / un.s)

    intensity = 2 * const.k_B * 1e-3 * un.K / wvlngth ** 2

    flux_density = 1e26 * intensity.to(un.W / (un.Hz * un.m ** 2))
    
    return flux_density.value * (( cellsize ) / distances)**2

def redshifts2frequencies(z):
    """The cosmological redshift (of signal) associated with each frequency"""
    return 1420e6 / (z + 1)

file_extension = ["bothBrokenOffsetData", "offsetData" , "brokenData"]
label_title = ["Broken + Offset", "Offset Only", "Broken Only"]
nkperp = 61
nkpar = 50
N_timestep = 1
N_realizations = 3000
N_components = [0, 20]
nu_min = 180

frac_true = True

# for extension_num in range(len(file_extension)):
# 	for iComponent in N_components:
# 		all_power = np.zeros([N_realizations, nkpar, nkperp-1])
# 		start = time.time()

# 		for iRealization in range(N_realizations):
# 			data = np.load("data/all_power/"+ file_extension[extension_num] + "/power_Ncomp%i_timesteps1_realization%i_FGEoR.npz" %(iComponent, iRealization))

# 			all_power[iRealization] = data["power"][50:]

# 		print("Done reading in",time.time()-start)

# 		np.savez("data/all_power/all_power_Ncomp%i_timesteps1_" %iComponent + file_extension[extension_num] +"_FGEoR", all_power = all_power, kperp = data["kperp"], kpar = data["kpar"])

power_default = np.load("data/all_power/power_Ncomp0_timesteps1_realization0_default_FGEoR.npz")["power"][50:]

kperp = np.load("data/all_power/all_power_Ncomp0_timesteps1_%s_FGEoR.npz" %file_extension[0])["kperp"] 
kpar = np.load("data/all_power/all_power_Ncomp0_timesteps1_%s_FGEoR.npz" %file_extension[0])["kpar"] 

# fig, ax = plt.subplots(2, 3, figsize=(10,8), sharex=True, sharey=True, gridspec_kw={"hspace":0.11, "wspace":0.03, "right":0.86, "left":0.09, "top":0.91, "bottom": 0.1})
# fig1, ax1 = plt.subplots(2, 3, figsize=(10,8), sharex=True, sharey=True,  gridspec_kw={"hspace":0.11, "wspace":0.03, "right":0.86, "left":0.09, "top":0.91, "bottom": 0.1})


# for extension_num in range(len(file_extension)):

# 	power0 = np.load("data/all_power/all_power_Ncomp0_timesteps1_%s_FGEoR.npz" %file_extension[extension_num])["all_power"] 
# 	power20 = np.load("data/all_power/all_power_Ncomp20_timesteps1_%s_FGEoR.npz" %file_extension[extension_num])["all_power"]

# 	if frac_true == False:
# 		frac = np.abs(power20 - power0) #/ power0
# 		print(np.min(frac), np.max(frac))

# 		img = ax[0, extension_num].imshow(np.log10(np.mean(frac, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=2)
# 		img = ax[1, extension_num].imshow(np.log10(np.std(frac, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=2)

# 		frac_default = np.abs(power20 - power_default) #/ power_default
# 		print(np.min(frac_default), np.max(frac_default))

# 		img1 = ax1[0, extension_num].imshow(np.log10(np.mean(frac_default, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=0.5)
# 		img1 = ax1[1, extension_num].imshow(np.log10(np.std(frac_default, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=0.5)

# 	else:
# 		frac = np.abs(power20 - power0) / power0
# 		print(file_extension[extension_num], "20 vs 0: ", np.min(np.mean(frac, axis=0)), np.max(np.mean(frac, axis=0)))

# 		img = ax[0, extension_num].imshow((np.mean(frac, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=1, cmap="Spectral")
# 		img = ax[1, extension_num].imshow((np.std(frac, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=1, cmap="Spectral")
		
# 		frac_default = np.abs(power20 - power_default) / power_default
# 		print("ideal: ", np.min(np.mean(frac_default, axis=0)), np.max(np.mean(frac_default, axis=0)))

# 		img1 = ax1[0, extension_num].imshow((np.mean(frac_default, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=5, vmax=10, cmap="Spectral")
# 		img1 = ax1[1, extension_num].imshow((np.std(frac_default, axis=0)), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=5, vmax=10, cmap="Spectral")

# 	ax[0, extension_num].set_xscale('log')
# 	ax[0, extension_num].set_yscale('log')
# 	ax[0, extension_num].set_title(label_title[extension_num]+"\n" + r"$\mu$", fontsize=18)

# 	ax[1, extension_num].set_title( r"$\sigma$", fontsize=18)
# 	ax[1, extension_num].set_xscale('log')
# 	ax[1, extension_num].set_yscale('log')

# 	ax1[0, extension_num].set_xscale('log')
# 	ax1[0, extension_num].set_yscale('log')
# 	ax1[0, extension_num].set_title(label_title[extension_num]+"\n" + r"$\mu$", fontsize=18)

# 	ax1[1, extension_num].set_title( r"$\sigma$", fontsize=18)
# 	ax1[1, extension_num].set_xscale('log')
# 	ax1[1, extension_num].set_yscale('log')


# 	ax[0, extension_num].tick_params(labelsize=15)
# 	ax[1, extension_num].tick_params(labelsize=15)

# 	ax1[0, extension_num].tick_params(labelsize=15)
# 	ax1[1, extension_num].tick_params(labelsize=15)

# ax_ = fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axes
# ax_.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# ax_.grid(False)

# ax_.set_xlabel(r"$r$", fontsize=20, labelpad=15)
# ax_.set_ylabel(r"$\eta$ [Hz$^{-1}$]", fontsize=20, labelpad=40)

# cax = fig.add_axes([0.87, 0.1, 0.03, 0.81])
# cax.tick_params(labelsize=15)

# if frac_true==False:
# 	cb = plt.colorbar(img, cax=cax).set_label(r"log$_{10}(|P_{20} - P_{0}|$) [Jy$^2$ Hz$^{2}]$", size=20) #"$|P_{20} - P_{0}$| / $P_{0}$", size=20)

# 	fig.savefig("figures/meanDifferencePowerNcomp_FGEoR") #meanFracDifferencePowerNcomp_FGEoR")
# else:
# 	cb = plt.colorbar(img, cax=cax).set_label(r"$|P_{20} - P_{0}$| / $P_{0}$", size=20)

# 	fig.savefig("figures/meanFracDifferencePowerNcomp_FGEoR")

# ax1_ = fig1.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axes
# ax1_.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# ax1_.grid(False)

# ax1_.set_xlabel(r"$r$", fontsize=20, labelpad=15)
# ax1_.set_ylabel(r"$\eta$ [Hz$^{-1}$]", fontsize=20, labelpad=40)

# cax1 = fig1.add_axes([0.87, 0.1, 0.03, 0.81])
# cax1.tick_params(labelsize=15)

# if frac_true== False:
# 	cb1 = plt.colorbar(img1, cax=cax1).set_label(r"log$_{10}(|P_{20} - P_{\rm ideal}|$) [Jy$^2$ Hz$^{2}]$", size=20) #"$|P_{20} - P_{\rm ideal}$| / $P_{\rm ideal}$", size=20)
# 	fig1.savefig("figures/meanDifferencePower_FGEoR") #meanFracDifferencePower_FGEoR")
# else:
# 	cb1 = plt.colorbar(img1, cax=cax1).set_label(r"$|P_{20} - P_{\rm ideal}$| / $P_{\rm ideal}$", size=20)
# 	fig1.savefig("figures/meanFracDifferencePower_FGEoR")

# plt.close("all")

# plot specific realization
realization = 1318

power0 = np.load("data/all_power/all_power_Ncomp0_timesteps1_%s_FGEoR.npz" %file_extension[0])["all_power"] 
power20 = np.load("data/all_power/all_power_Ncomp20_timesteps1_%s_FGEoR.npz" %file_extension[0])["all_power"]

fig, ax = plt.subplots(2,2, figsize=(9,8), sharex=True, sharey=True, gridspec_kw={"hspace":0.13, "wspace":0.1, "right":0.96, "left":0.09, "top":0.91, "bottom": 0.1})

img = ax[0,0].imshow((power0[realization]), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e11))
plt.colorbar(img, ax = ax[0,0]).set_label(r"mK$^2$ Mpc$^{3}$ $h^{-3}$", size=20)
ax[0,0].set_title(r"$P_{\rm true}$", fontsize=20)

img = ax[0,1].imshow((power20[realization]), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e11))
plt.colorbar(img, ax = ax[0,1]).set_label(r"mK$^2$ Mpc$^{3}$ $h^{-3}$", size=20)
ax[0,1].set_title(r"$P_{20}$", fontsize=20)

img = ax[1,0].imshow((np.abs(power20[realization] - power0[realization])), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e11))
plt.colorbar(img, ax = ax[1,0]).set_label(r"mK$^2$ Mpc$^{3}$ $h^{-3}$", size=20)
ax[1,0].set_title(r"$|P_{20}-P_{\rm true}|$", fontsize=20)

img = ax[1,1].imshow(np.abs(power20[realization] - power0[realization]) / power0[realization], origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=0.15, cmap="coolwarm")
plt.colorbar(img, ax = ax[1,1]) #.set_label(r"log$_{10}$ (P [mK$^2$ Mpc$^{3}$ $h^{-3}]$ )", size=20)
ax[1,1].set_title(r"$(|P_{20}-P_{\rm true}|)/P_{\rm true}$", fontsize=20)

for ii in range(2):
	for jj in range(2):
		ax[ii, jj].set_xscale('log')
		ax[ii, jj].set_yscale('log')
		ax[ii, jj].tick_params(labelsize=15)

ax_ = fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
ax_.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax_.grid(False)

ax_.set_xlabel(r"$k_\perp$ [$h$ Mpc$^{-1}$]", fontsize=22)
ax_.set_ylabel(r"$k_\parallel$ [$h$ Mpc$^{-1}$]", fontsize=22, labelpad=20)

fig.savefig("figures/delayPower_realization%i" %realization + file_extension[0])

fig1, ax1 = plt.subplots(2,2, figsize=(9,8), sharex=True, sharey=True, gridspec_kw={"hspace":0.13, "wspace":0.1, "right":0.96, "left":0.09, "top":0.91, "bottom": 0.1})

img1 = ax1[0,0].imshow((power0[realization]), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e11))
plt.colorbar(img1, ax = ax1[0,0]).set_label(r"mK$^2$ Mpc$^{3}$ $h^{-3}$", size=20)
ax1[0,0].set_title(r"$P_{\rm true}$", fontsize=20)

img1 = ax1[0,1].imshow((power_default), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e11))
plt.colorbar(img1, ax = ax1[0,1]).set_label(r"mK$^2$ Mpc$^{3}$ $h^{-3}$", size=20)
ax1[0,1].set_title(r"$P_{\rm ideal}$", fontsize=20)

img1 = ax1[1,0].imshow((np.abs(power_default - power0[realization])), origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), norm=colors.LogNorm(vmin=1, vmax=1e11))
plt.colorbar(img1, ax = ax1[1,0]).set_label(r"mK$^2$ Mpc$^{3}$ $h^{-3}$", size=20)
ax1[1,0].set_title(r"$|P_{\rm ideal}-P_{\rm true}|$", fontsize=20)

img1 = ax1[1,1].imshow(np.abs(power_default - power0[realization]) / power0[realization], origin="lower", extent=(kperp[0], kperp[-1], kpar[51], kpar[-1]), vmin=0, vmax=1, cmap="coolwarm")
plt.colorbar(img1, ax = ax1[1,1]) #.set_label(r"log$_{10}$ (P [mK$^2$ Mpc$^{3}$ $h^{-3}]$ )", size=20)
ax1[1,1].set_title(r"$(|P_{\rm ideal} - P_{\rm true}|)/P_{\rm true}$", fontsize=20)

for ii in range(2):
	for jj in range(2):
		ax1[ii, jj].set_xscale('log')
		ax1[ii, jj].set_yscale('log')
		ax1[ii, jj].tick_params(labelsize=15)

ax1_ = fig1.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
ax1_.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax1_.grid(False)

ax1_.set_xlabel(r"$k_\perp$ [$h$ Mpc$^{-1}$]", fontsize=22)
ax1_.set_ylabel(r"$k_\parallel$ [$h$ Mpc$^{-1}$]", fontsize=22, labelpad=20)

fig1.savefig("figures/idealDelayPower_realization%i" %realization + file_extension[0])

raise SystemExit

box = np.load("data/box_nu100_%iMHz.npz" %(nu_min))["box"][:,:,0]
fig, ax = plt.subplots(1,1, figsize=(10,8), sharex=True, sharey=True, gridspec_kw={"hspace":0.11, "wspace":0.03, "right":0.86, "left":0.09, "top":0.91, "bottom": 0.1})

im = ax.imshow(box, extent=(-10,10,-10,10))
ax.set_xlabel(r"$l$ [degrees]", fontsize=25)
ax.set_ylabel(r"$m$ [degrees]", fontsize=25)
ax.tick_params(labelsize=15)
cax = fig.add_axes([0.82, 0.1, 0.03, 0.81])
cax.tick_params(labelsize=15)
plt.colorbar(im, cax=cax).set_label(r"$I(l, m)$ [Jy sr$^{-1}]$", size=25)
fig.savefig("figures/fg_sky")

# read the EoR lightcone data and change to Jy
data = np.load("../SKAdataChallenge/data/lightcone_%iMHz_2.0degrees.npz" %nu_min)

fig, ax = plt.subplots(1,1, figsize=(10,8), sharex=True, sharey=True, gridspec_kw={"hspace":0.11, "wspace":0.03, "right":0.86, "left":0.09, "top":0.91, "bottom": 0.1})

im = ax.imshow(data["lightcone"][:,:, 0], extent=(-1, 1, -1, 1))
ax.set_xlabel(r"$l$ [degrees]", fontsize=25)
ax.set_ylabel(r"$m$ [degrees]", fontsize=25)
ax.tick_params(labelsize=15)
cax = fig.add_axes([0.82, 0.1, 0.03, 0.81])
cax.tick_params(labelsize=15)
plt.colorbar(im, cax=cax).set_label(r"$T_B$ [mK]", size=25)
fig.savefig("figures/eor_sky")
