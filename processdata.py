import numpy as np
import pandas as pd
import h5py
import os
from collections import OrderedDict
from scipy.stats import binned_statistic

import dask
import dask.array as da

"""
Funtions for analyzing XPP small data
"""
ADU_per_photon = 195.
min_ADU = 170.
I0_key = 'ipm3/sum'

folder = '/reg/d/psdm/xpp/xpplp7515/hdf5/smalldata'

ipm_threshold  = -np.inf
eV_per_pix = 1 # 0.0082 # from earlier data

def get_Motors(f, Motors=OrderedDict()):
    """ Get dictionary defining which motors to load.
    This is of form.
    {<key in dictionary> : <path in h5 file>}
    Extra values scan be added by specifing motors as OrderedDic
    """
    # Load droplets

    Motors.update(OrderedDict(
        x ='epix/droplet_X',
        y = 'epix/droplet_Y',
        ADU = 'epix/droplet_adu',
        Npix = 'epix/droplet_npix',
        ipm2 = 'ipm2/sum',
        ipm3 = 'ipm3/sum',
        jitter = 'epics/lxt_ttc',
        laser = 'lightStatus/laser',
        xray = 'lightStatus/xray'
                 )
                 )
    Motors.update({'diodeU': 'diodeU/channels', 'delay_correction': 'phase_cav/fit_time_2'})
    # Load user motors
    user_motors = list(f['epicsUser'].keys())
    for motor in user_motors:
        Motors[motor] = 'epicsUser/'+ motor

    # Load epics motors
    epics_motors = list(f['epics'].keys())
    epics_motors_used = [key for key in epics_motors if ('robot' not in key and 'slit' not in key)]
    for motor in epics_motors_used:
        Motors[motor] = 'epics/' + motor

    # Load scan motor
    if len(f['scan'].keys()) >=3:
        first_key = 'scan/' + list(f['scan'].keys())[0]
        Motors[first_key] = first_key

    return Motors

def get_data(run, Motors=OrderedDict(), fpath=None):
    """ Load data into ordered dictionary. Motors is a dictionary of
    {<key in dictionary> : <path in h5 file>} for pulling extra entries
    path can be specified explicitly to access other data"""
    if fpath == None:
        fpath = os.path.join(folder, 'xpplp7515_Run{:03d}.h5'.format(run))
    f = h5py.File(fpath)
    allIPM = f[I0_key].value
    goodIPM = (allIPM>ipm_threshold)

    if Motors == OrderedDict():
        Motors.update(get_Motors(f))

    d = OrderedDict()
    for key, name in Motors.items():
        try:
            d[key] = f[name].value[goodIPM]
        except KeyError:
            print("key {} not found!".format(key))

    try:
        d['corrected_delay'] = d['scan/lxt']*1.*10**12 + d['delay_correction']
    except KeyError:
        pass

    return d

def combine(list_of_d):
    """ combine several d objects from different runs into single object. """
    d_combined = {}
    first = list_of_d[0]
    for key in first.keys():
        if len(first[key].shape) == 1:
            d_combined[key] = np.hstack((d[key] for d in list_of_d))
        else:
            d_combined[key] = np.vstack((d[key] for d in list_of_d))

    return d_combined

def clean_data(d, selectkey='ADU', minkey=min_ADU, maxkey=1000, minx=-np.inf, maxx=np.inf, miny=-np.inf, maxy=np.inf):
    """ Select droplets based on their X, Y location and their ADU Npix etc. """
    choose_events = np.logical_and.reduce((d[selectkey] > minkey,
                                   d[selectkey] <= maxkey,
                                   d['x'] > minx,
                                   d['x'] <= maxx,
                                   d['y'] > miny,
                                   d['y'] <= maxy
                                  ))
    print("Keeping {:.2f}%".format(100*np.sum(choose_events)/choose_events.size))
    d_clean = OrderedDict()
    for key, item in d.items():
        try:
            d_clean[key] = d[key][choose_events]
        except IndexError:
            d_clean[key] = d[key]
    return d_clean

def select_scan(d, motorkey, motorPos, tolerance=1.e-5):
    '''
    Output a list of OrderedDict. Each containing shots from a single delay.
    '''
    d_scan = []
    for motorval in motorPos:
        goodshots = np.logical_and(d[motorkey]>motorval-tolerance, d[motorkey]<=motorval+tolerance)
        tempd = OrderedDict()
        for key in d.keys():
            tempd[key] = d[key][goodshots]
        d_scan.append(tempd)
    return motorPos, d_scan

def select(d, motorkey, motorval, tolerance=1.e-5):
    '''
    Return a selected version of d
    '''
    goodshots = np.logical_and(d[motorkey]>motorval-tolerance, d[motorkey]<=motorval+tolerance)
    d_select = OrderedDict()
    for key in d.keys():
        d_select[key] = d[key][goodshots]
    return d_select

def bin_rixs_slope(x, y, y_edges, ADUs, elPix = 300, slope=0):
	""" sum ADU values along x within y ranges (defined by edges)"""
	y_cens = (y_edges[0:-1] + y_edges[0:-1])/2
	E = (y_cens - elPix) * eV_per_pix
	I, _ = np.histogram(y - x*slope, bins=y_edges, weights=ADUs)

	return E, I

def bin_rixs(y, y_edges, ADUs, elPix = 300):
	""" sum ADU values along x within y ranges (defined by edges)"""
	y_cens = (y_edges[0:-1] + y_edges[0:-1])/2
	E = (y_cens - elPix) * eV_per_pix
	I, _ = np.histogram(y, bins=y_edges, weights=ADUs)
	return E, I

def bin_edges_centers(minvalue, maxvalue, binsize):
    """Make bin edges and centers for use in histogram
    The rounding of the bins edges is such that all bins are fully populated.
    Parameters
    -----------
    minvalue/maxvalue : array/array
        minimun/ maximum
    binsize : float (usually a whole number)
        difference between subsequnt points in edges and centers array
    Returns
    -----------
    edges : array
        edge of bins for use in np.histrogram
    centers : array
        central value of each bin. One shorter than edges
    """
    edges = binsize * np.arange(minvalue//binsize + 1, maxvalue//binsize)
    centers = (edges[:-1] + edges[1:])/2
    return edges, centers

def make_image(d):
    """ Construct a 2D image array by histograming the x, y, ADU data"""
    minx = np.min(d['x'])
    maxx = np.max(d['x'])
    miny = np.min(d['y'])
    maxy = np.max(d['y'])
    x_edges, x_centers = bin_edges_centers(minx, maxx, 1)
    y_edges, y_centers = bin_edges_centers(miny, maxy, 1)
    M, _, _ = np.histogram2d(d['y'], d['x'],  bins=(y_edges, x_edges), weights=d['ADU'])
    return x_centers, y_centers, M

def get_d(run, ADU_per_photon=190, minx = 140, maxx = 240, miny=100, maxy=380):
    """Wrapper to pull everything using the traditional method"""
    print("Run = {}".format(run))
    d = get_data(run)
    d_clean = clean_data(d, selectkey='ADU', minkey=170, maxkey=1000, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    return d_clean

def get_RIXS_multi(runs, ADU_per_photon=190, minADU=170, maxADU=1000, minx=140, maxx=240, miny=100, maxy=380, binsize=1, elPix=0, laser_select=None, xray_select=None, slope=0):
    """Fast get of RIXS data"""
    d = combine([get_data(run) for run in runs])
    if laser_select != None:
        d = select(d, 'laser', laser_select)
    if xray_select != None:
        d = select(d, 'xray', xray_select)

    d = clean_data(d, selectkey='ADU', minkey=minADU, maxkey=maxADU, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    x, y, M = make_image(d)
    pixels_edges, pixel_centers = bin_edges_centers(miny, maxy, binsize)
    #E, I = bin_rixs(d['y'], pixels_edges, d['ADU']/ADU_per_photon, elPix = elPix)
    E, I = bin_rixs_slope(d['x'], d['y'], pixels_edges, d['ADU']/ADU_per_photon, elPix = elPix, slope=slope)
    M /= ADU_per_photon
    X, Y = np.meshgrid(x, y)

    RIXS = OrderedDict(d=d, E=E, I=I, M=M, X=X, Y=Y, run=runs)
    return RIXS

def get_RIXS(run, ADU_per_photon=190, minADU=170, maxADU=1000, minx=140, maxx=240, miny=100, maxy=380, binsize=1, elPix=0, laser_select=None, xray_select=None):
    """Fast get of RIXS data"""
    d = get_data(run)
    if laser_select != None:
        d = select(d, 'laser', laser_select)
    if xray_select != None:
        d = select(d, 'xray', xray_select)

    d = clean_data(d, selectkey='ADU', minkey=minADU, maxkey=maxADU, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    x, y, M = make_image(d)
    pixels_edges, pixel_centers = bin_edges_centers(miny, maxy, binsize)
    E, I = bin_rixs(d['y'], pixels_edges, d['ADU']/ADU_per_photon, elPix = elPix)
    M /= ADU_per_photon
    X, Y = np.meshgrid(x, y)

    RIXS = OrderedDict(d=d, E=E, I=I, M=M, X=X, Y=Y, run=run)
    return RIXS

# def get_d_fast(run, ADU_per_photon=190, minADU=170, maxADU=1000, minx=140, maxx=240, miny=100, maxy=380, laser_select=None, xray_select=None):
#     """Fast pull of minimum d parallelized using dask"""
#     print("Run = {}".format(run))
#     fpath = os.path.join(folder, 'xpplp7515_Run{:03d}.h5'.format(run))
#     f = h5py.File(fpath)
#
#     ipm2 = da.from_array(f['ipm2/sum'], chunks=(2e6))
#     ipm3 = da.from_array(f['ipm3/sum'], chunks=(2e6))
#     laser = da.from_array(f['lightStatus/laser'], chunks=(2e6))
#     xray = da.from_array(f['lightStatus/xray'], chunks=(2e6))
#
#     chunks = (2000, 2000)
#     x = da.from_array(f['epix/droplet_X'], chunks=chunks)
#     y = da.from_array(f['epix/droplet_Y'],  chunks=chunks)
#     ADU = da.from_array(f['epix/droplet_adu'], chunks=chunks)
#
#     if laser_select != None:
#         print("laser select not implemented!!!!! As much slower")
#         #pick = (laser == laser_select).compute()
#         #ipm2 = ipm2[pick]
#         #ipm3 = ipm3[pick]
#         #xray = xray[pick]
#         #laser = laser[pick]
#         #x = x[pick,:]
#         #y = y[pick,:]
#         #ADU = ADU[pick,:]
#     if xray_select != None:
#         print("xray select not implemented!!!!! As much slower")
#
#     choose = (x>minx) & (x<=maxx) & (y>miny) & (y<=maxy) & (ADU>minADU) & (ADU<=maxADU)
#
#
#
#     results_list = dask.compute(x[choose], y[choose], ADU[choose], ipm2, ipm3, laser, xray)
#
#     return OrderedDict([key, val] for key, val in zip(['x', 'y', 'ADU', 'imp2', 'imp3', 'laser', 'xray'], results_list))

#RIXS_list = [RIXS, RIXS, RIXS, RIXS]
#
#def combine_RIXS(RIXS_list):
#    RIXStot = OrderedDict()
#    for key in ['I', 'M']:
#        RIXStot[key] = sum(RIXS[key] for RIXS in RIXS_list)
#    for key in RIXS_list[0].keys():
#        if key not in ['I', 'M', 'run', 'd']:
#            RIXStot[key] = RIXS_list[0][key]
#
#    RIXStot['run'] = [RIXS['run'] for RIXS in RIXS_list]
#    return RIXStot
#
#RIXStot = combine_RIXS(RIXS_list)

#def get_d_fast(run, ADU_per_photon=190, minADU=170, maxADU=1000, minx=140, maxx=240, miny=100, maxy=380):
#    """Fast pull of minimum d parallelized using dask"""
#    print("Run = {}".format(run))
#    fpath = os.path.join(folder, 'xpplp7515_Run{:03d}.h5'.format(run))
#    f = h5py.File(fpath)
#
#    chunks = (2000, 2000)
#    ipm = da.from_array(f['epix/droplet_X'], chunks=chunks)
#    x = da.from_array(f['epix/droplet_X'], chunks=chunks)
#    y = da.from_array(f['epix/droplet_Y'],  chunks=chunks)
#    ADU = da.from_array(f['epix/droplet_adu'], chunks=chunks)
#    choose = (x>minx) & (x<=maxx) & (y>miny) & (y<=maxy) & (ADU>minADU) & (ADU<=maxADU)
#
#    ipm2 = da.from_array(f['ipm2/sum'], chunks=(2e6))
#    ipm3 = da.from_array(f['ipm3/sum'], chunks=(2e6))
#
#    laser = da.from_array(f['lightStatus/laser'], chunks=(2e6))
#    xray = da.from_array(f['lightStatus/xray'], chunks=(2e6))
#
#    results_list = dask.compute(x[choose], y[choose], ADU[choose], ipm2, ipm3, laser, xray)
#
#    return OrderedDict([key, val] for key, val in zip(['x', 'y', 'ADU', 'imp2', 'imp3', 'laser', 'xray'], results_list))


# def get_RIXS_fast(run, ADU_per_photon=190, minADU=170, maxADU=1000, minx=140, maxx=240, miny=100, maxy=380, binsize=1, elPix=0, laser_select=None, xray_select=None):
#     """Fast get of RIXS data"""
#     try:
#         d = get_d_fast(run, minADU=minADU, maxADU=maxADU, ADU_per_photon=ADU_per_photon,
#                        minx=minx, maxx=maxx, miny=miny, maxy=maxy,
#                        laser_select=laser_select, xray_select=xray_select)
#     except TypeError:
#         d = combine([get_d_fast(run_val, minADU=minADU, maxADU=maxADU, ADU_per_photon=ADU_per_photon,
#                                 minx=minx, maxx=maxx, miny=miny, maxy=maxy,
#                                 laser_select=laser_select, xray_select=xray_select)
#                      for run_val in run])
#
#     if laser_select != None:
#         print("Cannot do this selection in fast version!!")
#     if xray_select != None:
#         dprint("Cannot do this selection in fast version!!")
#     x, y, M = make_image(d)
#     pixels_edges, pixel_centers = bin_edges_centers(miny, maxy, binsize)
#     E, I = bin_rixs(d['y'], pixels_edges, d['ADU']/ADU_per_photon, elPix = elPix)
#     M /= ADU_per_photon
#     X, Y = np.meshgrid(x, y)
#
#     RIXS = OrderedDict(d=d, E=E, I=I, M=M, X=X, Y=Y, run=run)
#     return RIXS
