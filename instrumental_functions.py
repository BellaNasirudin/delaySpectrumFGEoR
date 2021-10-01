import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo
from powerbox import LogNormalPowerBox, PowerBox
from powerbox.dft import fft, fftfreq
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import multiprocessing as mp

def interpolate_frequencies(data, freqs, linFreqs, uv_range=100, new_ncells = None):

    if (freqs[0] > freqs[-1]):
        freqs = freqs[::-1]
        data = np.flip(data, 2)

    ncells = np.shape(data)[0]
    # Create the xy data
    xy = np.linspace(-uv_range / 2., uv_range / 2., ncells)

    # generate the interpolation function
    func = RegularGridInterpolator([xy, xy, freqs], data, bounds_error=False, fill_value=0)

    if new_ncells!= None:
        # Create the xy data
        xy = np.linspace(-uv_range / 2., uv_range / 2., new_ncells)
        ncells = new_ncells

    # Create a meshgrid to interpolate the points
    XY, YX, LINFREQS = np.meshgrid(xy, np.flip(xy), linFreqs)

    # Flatten the arrays so the can be put into pts array
    XY = XY.flatten()
    YX = YX.flatten()
    LINFREQS = LINFREQS.flatten()

    # Create the points to interpolate
    numpts = XY.size
    pts = np.zeros([numpts, 3])
    pts[:, 0], pts[:, 1], pts[:, 2] = XY, YX, LINFREQS

    # Interpolate the points
    interpData = func(pts)

    # Reshape the data
    interpData = interpData.reshape(ncells, ncells, len(linFreqs))

    return interpData
    
def sigma(frequencies, tile_diameter = 35):
    "The Gaussian beam width at each frequency"
    epsilon = 0.42  # scaling from airy disk to Gaussian
    return ((epsilon * const.c) / (frequencies / un.s *  tile_diameter * un.m)).to(un.dimensionless_unscaled).value

def gaussian_beam(sky_size, frequencies, n_cells, min_attenuation = 5e-7, tile_diameter = 35):
    """
    Generate a frequency-dependent Gaussian beam attenuation across the sky per frequency.
    Parameters
    ----------
    n_cells : int
        Number of cells in the sky grid.
    sky_size : float
        The extent of the sky in lm.
    Returns
    -------
    attenuation : (ncells, ncells, nfrequencies)-array
        The beam attenuation (maximum unity) over the sky.
    
    """
    sky_coords = np.linspace(- sky_size / 2,  sky_size / 2,  n_cells)
    
    # Create a meshgrid for the beam attenuation on sky array
    L, M = np.meshgrid(np.sin(sky_coords), np.flip(np.sin(sky_coords)))

    sigma_beam = sigma(frequencies, tile_diameter)

    attenuation = np.exp( np.outer(-(L ** 2 + M ** 2), 1. / (sigma_beam ** 2)).reshape(
            ( n_cells,  n_cells, len(frequencies))))
    
    attenuation[attenuation<min_attenuation] = 0
    
    return attenuation

def image_to_uv(sky, L_box):
    """
    Transform a box from image plan to UV plane.
    Parameters
    ----------
    sky : (ncells, ncells, nfreq)-array
        The frequency-dependent sky brightness (in arbitrary units)
    L_box : float
        The size of the box in radians.
    Returns
    -------
    uvsky : (ncells, ncells, nfreq)-array
        The UV-plane representation of the sky. Units are units of the sky times radians.
    uv_scale : list of two arrays.
        The u and v co-ordinates of the uvsky, respectively. Units are inverse of L.
    """
    print("Converting to UV space...")

    ft, uv_scale = fft(sky, L_box, axes=(0, 1), a=0, b=2 * np.pi)

    return ft, uv_scale

def sample_onto_baselines(uvplane, uv, baselines, frequencies):
    """
    Sample a gridded UV sky onto a set of baselines.
    Sampling is done via linear interpolation over the regular grid.
    Parameters
    ----------
    uvplane : (ncells, ncells, nfreq)-array
        The gridded UV sky, in Jy.
    uv : list of two 1D arrays
        The u and v coordinates of the uvplane respectively.
    baselines : (N,2)-array
        Each row should be the (x,y) co-ordinates of a baseline, in metres.
    frequencies : 1D array
        The frequencies of the uvplane.
    Returns
    -------
    vis : complex (N, nfreq)-array
         The visibilities defined at each baseline.
    """

    frequencies = frequencies / un.s
    vis = np.zeros((len(baselines), len(frequencies)), dtype=np.complex128)

    print("Sampling the data onto baselines...")

    for i, ff in enumerate(frequencies):
        lamb = const.c / ff.to(1 / un.s)
        arr = np.zeros(np.shape(baselines))
        arr[:, 0] = (baselines[:, 0] / lamb).value
        arr[:, 1] = (baselines[:, 1] / lamb).value

        real = np.real(uvplane[:, :, i])
        imag = np.imag(uvplane[:, :, i])

        f_real = RectBivariateSpline(uv[0], uv[1], real)
        f_imag = RectBivariateSpline(uv[0], uv[1], imag)

        FT_real = f_real(arr[:, 0], arr[:, 1], grid=False)
        FT_imag = f_imag(arr[:, 0], arr[:, 1], grid=False)

        vis[:, i] = FT_real + FT_imag * 1j

    return vis

def _sample_onto_baselines_buff(ncells,nfreqall, nfreqoffset,uvplane, uv, baselines, frequencies, vis_buff_real, vis_buff_imag):
    """
    Sample a gridded UV sky onto a set of baselines.
    Sampling is done via linear interpolation over the regular grid.
    Parameters
    ----------
    uvplane : (ncells, ncells, nfreq)-array
        The gridded UV sky, in Jy.
    uv : list of two 1D arrays
        The u and v coordinates of the uvplane respectively.
    baselines : (N,2)-array
        Each row should be the (x,y) co-ordinates of a baseline, in metres.
    frequencies : 1D array
        The frequencies of the uvplane.
    Returns
    -------
    vis : complex (N, nfreq)-array
         The visibilities defined at each baseline.
    """

    vis_real = np.frombuffer(vis_buff_real).reshape(baselines.shape[0],len(frequencies))
    vis_imag = np.frombuffer(vis_buff_imag).reshape(baselines.shape[0],len(frequencies))

    frequencies = frequencies / un.s
    
    print("Sampling the data onto baselines...")

    for i, ff in enumerate(frequencies):
        lamb = const.c / ff.to(1 / un.s)
        arr = np.zeros(np.shape(baselines))
        arr[:, 0] = (baselines[:, 0] / lamb).value
        arr[:, 1] = (baselines[:, 1] / lamb).value
        np.savetxt("data/baselines_%i" %i, arr)
        real = uvplane.real[:, :, i+nfreqoffset]
        imag = uvplane.imag[:, :, i+nfreqoffset]
        
        f_real = RectBivariateSpline(uv[0], uv[1], real)
        f_imag = RectBivariateSpline(uv[0], uv[1], imag)
        
        FT_real = f_real(arr[:, 0], arr[:, 1], grid=False)
        FT_imag = f_imag(arr[:, 0], arr[:, 1], grid=False)
        vis_real[:, i] = FT_real
        vis_imag[:, i] =  FT_imag


def sample_onto_baselines_parallel(uvplane, uv, baselines, frequencies, nparallel=1):

    #Find out the number of frequencies to process per thread
    nfreq = len(frequencies)
    ncells = uvplane.shape[0]
    numperthread = int(np.ceil(nfreq/nparallel))
    offset = 0
    nfreqstart = np.zeros(nparallel,dtype=int)
    nfreqend = np.zeros(nparallel,dtype=int)
    infreq = np.zeros(nparallel,dtype=int)
    for i in range(nparallel):
        nfreqstart[i] = offset
        nfreqend[i] = offset + numperthread

        if(i==nparallel-1):
            infreq[i] = nfreq - offset
        else:
            infreq[i] = numperthread

        offset+=numperthread

    # Set the last process to the number of frequencies
    nfreqend[-1] = nfreq
    processes = []
    vis_real = []
    vis_imag = []

    vis = np.zeros([baselines.shape[0],nfreq],dtype=np.complex128)

    #Lets split this array up into chunks
    for i in range(nparallel):

        #Get the buffer that contains the memory
        vis_buff_real = mp.RawArray(np.sctype2char(vis.real),vis[:,nfreqstart[i]:nfreqend[i]].size)
        vis_buff_imag = mp.RawArray(np.sctype2char(vis.real),vis[:,nfreqstart[i]:nfreqend[i]].size)

        vis_real.append(vis_buff_real)
        vis_imag.append(vis_buff_imag)

        processes.append(mp.Process(target=_sample_onto_baselines_buff,args=(ncells,nfreq,nfreqstart[i],uvplane, uv, baselines, frequencies[nfreqstart[i]:nfreqend[i]], vis_buff_real,vis_buff_imag) ))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    for i in range(nparallel):
        vis.real[:,nfreqstart[i]:nfreqend[i]] = np.frombuffer(vis_real[i]).reshape(baselines.shape[0],nfreqend[i] - nfreqstart[i])
        vis.imag[:,nfreqstart[i]:nfreqend[i]] = np.frombuffer(vis_imag[i]).reshape(baselines.shape[0],nfreqend[i] - nfreqstart[i])

    return vis

def get_baselines(x, y):
    """
    From a set of antenna positions, determine the non-autocorrelated baselines.
    Parameters
    ----------
    x, y : 1D arrays of the same length.
        The positions of the arrays (presumably in metres).
    Returns
    -------
    baselines : (n_baselines,2)-array
        Each row is the (x,y) co-ordinate of a baseline, in the same units as x,y.
    """
    # ignore up.
    ind = np.tril_indices(len(x), k=-1)
    Xsep = np.add.outer(x, -x)[ind]
    Ysep = np.add.outer(y, -y)[ind]

    # Remove autocorrelations
    zeros = np.logical_and(Xsep.flatten() == 0, Ysep.flatten() == 0)

    return np.array([Xsep.flatten()[np.logical_not(zeros)], Ysep.flatten()[np.logical_not(zeros)]]).T

def get_baselines_rotation(pos_file, tot_daily_obs_time = 6, int_time = 600, declination=-27, RA_pointing = 0):
    """
    From a set of antenna positions, determine the non-autocorrelated baselines with Earth rotation synthesis, assuming
    a flat sky.
    Parameters
    ----------
    pos_file : 2D array.
        The (x, y ) positions of the arrays (presumably in metres).
    tot_daily_obs_time: float
        The total observation time per day in hours.
    int_time:
        The interval between snapshots of the sky in seconds.
    Returns
    -------
    new_baselines : (n_baselines,2)-array
        Each row is the (x,y) co-ordinate of a baseline, in the same units as x,y.
    """
    number_of_snapshots = int(tot_daily_obs_time * 60 * 60 / int_time)

    new_baselines = np.zeros(( number_of_snapshots*len(pos_file), 2))

    for ii in range(number_of_snapshots):
        new_baselines[ii*len(pos_file):(ii+1)*len(pos_file),:] =  earth_rotation_synthesis(pos_file, ii, int_time, declination=declination, RA_pointing = RA_pointing)

    return new_baselines # only return the x,y part


def earth_rotation_synthesis(Nbase, slice_num, int_time, declination=-26., RA_pointing = 0):
    """
    The rotation of the earth over the observation times makes changes the part of the 
    sky measured by each antenna.
    Based on https://science.nrao.edu/science/meetings/2016/15th-synthesis-imaging-workshop/SISS15Advanced.pdf
    Parameters
    ----------
    Nbase       : ndarray
        The array containing all the ux,uy,uz values of the antenna configuration.
    slice_num   : int
        The number of the observed slice after each of the integration time.
    int_time    : float
        The time after which the signal is recorded (in seconds).
    declination : float
        Refers to the lattitute where telescope is located 
        (in degrees). Default: -27
    RA_pointing : float
        Refers to the RA of the observation
        (in hours!). Default: 0
    Returns
    -------
    new_Nbase   : ndarray
        It is the new Nbase calculated for the rotated antenna configurations.
    """

    # change everything in degree to radian because numpy does things in radian
    deg_to_rad = np.pi / 180.

    delta = deg_to_rad * declination

    one_hour = 15.0 * deg_to_rad # the rotation in radian after an hour

    # multiply by the total observation time and number of slices
    # also offset by the RA pointing
    HA    =  one_hour * (slice_num - 1) * int_time / (60 * 60) + RA_pointing * 15 * deg_to_rad
    
    new_Nbase = np.zeros((len(Nbase),2))
    new_Nbase[:,0] = np.sin(HA) * Nbase[:,0] + np.cos(HA) * Nbase[:,1]
    new_Nbase[:,1] = -1.0 * np.sin(delta) * np.cos(HA) * Nbase[:,0] + np.sin(delta) * np.sin(HA) * Nbase[:,1]

    return new_Nbase


def thermal_variance_baseline(instrumental_frequencies, Tsys, effective_collecting_area, integration_time):
    """
    The thermal variance of each baseline (assumed constant across baselines/times/frequencies.
    Equation comes from Trott 2016 (from Morales 2005)
    """
    df =  instrumental_frequencies[1] -  instrumental_frequencies[0]

    sigma = 2 * 1e26 * const.k_B.value *  Tsys /  effective_collecting_area / np.sqrt(
        df *  integration_time)

    return (sigma ** 2)


def add_thermal_noise(visibilities, instrumental_frequencies, effective_collecting_area, Tsys, integration_time):
    """
    Add thermal noise to each visibility.
    Parameters
    ----------
    visibilities : (n_baseline, n_freq)-array
        The visibilities at each baseline and frequency.
    frequencies : (n_freq)-array
        The frequencies of the observation.
    
    beam_area : float
        The area of the beam (in sr).
    delta_t : float, optional
        The integration time.
    Returns
    -------
    visibilities : array
        The visibilities at each baseline and frequency with the thermal noise from the sky.
    """
    
    print("Adding thermal noise...")
    rl_im = np.random.normal(0, 1, (2,) + visibilities.shape)

    # NOTE: we divide the variance by two here, because the variance of the absolute value of the
    #       visibility should be equal to thermal_variance_baseline, which is true if the variance of both
    #       the real and imaginary components are divided by two.
    return visibilities + np.sqrt(thermal_variance_baseline(instrumental_frequencies, Tsys, effective_collecting_area, integration_time) / 2) * (rl_im[0, :] + rl_im[1, :] * 1j)

def add_instrument_rotation(lightcone, frequencies, baselines, number_of_snapshots, beam, sky_size, tot_daily_obs_time, int_time, effective_collecting_area=300, Tsys=300, integration_time=100, nparallel=1, add_beam=True):

    baselines = get_baselines_rotation(baselines, tot_daily_obs_time = tot_daily_obs_time, int_time = int_time)
    L = int(len(baselines) / number_of_snapshots)

    all_visibilities = np.zeros((number_of_snapshots, L, len(frequencies)), dtype=np.complex128)

    for ii in range(number_of_snapshots):
        if add_beam==True:
            if number_of_snapshots== 1:
                lightcone_new = lightcone * beam
            else:
                lightcone_new = lightcone * beam[ii]
        else:
            lightcone_new = lightcone
        
        # Fourier Transform over the (l,m) dimension 
        lightcone_new, uv = image_to_uv(lightcone_new, sky_size)

        # baselines sampling        
        if(nparallel==1):
            all_visibilities[ii] = sample_onto_baselines(lightcone_new, uv, baselines[ii*L:(ii+1)*L], frequencies)
        else:
            all_visibilities[ii] = sample_onto_baselines_parallel(lightcone_new, uv, baselines[ii*L:(ii+1)*L], frequencies)

    # visibilities_noise = add_thermal_noise(all_visibilities.reshape((number_of_snapshots * L, len(frequencies))), frequencies, effective_collecting_area=effective_collecting_area, Tsys=Tsys, integration_time=integration_time)
  
    return all_visibilities.reshape((number_of_snapshots * L, len(frequencies))), baselines