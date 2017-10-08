import numpy as np
import h5py
import os
from collections import OrderedDict

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
        ipm = 'ipm3/sum',
        jitter = 'epics/lxt_ttc',
        laser = 'lightStatus/laser',
        xray = 'lightStatus/xray'
                 )
                 )
    
    # Load user motors
    user_motors = list(f['epicsUser'].keys())
    for motor in user_motors:
        Motors[motor] = 'epicsUser/'+ motor
    
    # Load epics motors
    epics_motors = list(f['epics'].keys())
    epics_motors_used = [key for key in epics_motors if ('robot' not in key and 'slit' not in key)]
    for motor in epics_motors_used:
        Motors[motor] = 'epics/' + motor
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
        d[key] = f[name].value[goodIPM]
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


# SOME OLD PATHS TO FIELDS HERE
#folder = '/reg/d/psdm/xpp/xppl1316/res/littleData/ftc'
#fpath = os.path.join(folder, 'ldat_xppl1316_Run{}.h5'.format(run))
#
#    d = OrderedDict(
#        x =f['epix/droplet_X'].value[goodIPM],
#        y = f['epix/droplet_Y'].value[goodIPM],
#        ADU = f['epix/droplet_adu'].value[goodIPM],
#        Npix = f['epix/droplet_npix'].value[goodIPM],
#        ipm = f['ipm3/sum'].value[goodIPM],
#        jitter = f['epics/lxt_ttc'].value[goodIPM],
#        laser = f['lightStatus/laser'].value[goodIPM],
#        xray = f['lightStatus/xray'].value[goodIPM]
#                 )
#
#def get_data(run, ScanMotors={}, fpath=None):
#    if fpath == None:
#        #fpath = os.path.join(folder, 'ldat_xppl1316_Run{}.h5'.format(run))
#        fpath = os.path.join(folder, 'xpplp7515_Run{}.h5'.format(str(run).zfill(3)))
#    f = h5py.File(fpath)
#    allIPM = f['ipm3/sum'].value
#    goodIPM = (allIPM>ipm_threshold)
#
#    d = OrderedDict(
#        #x =f['epix/dropletsX'].value[goodIPM],
#        #y = f['epix/dropletsY'].value[goodIPM],
#        #ADU = f['epix/dropletsAdu'].value[goodIPM],
#        #Npix = f['epix/dropletsNpix'].value[goodIPM],
#        #ipm = f['ipm3/sum'].value[goodIPM],
#        #jitter = f['epics/lxt_ttc'].value[goodIPM],
#        #laser = f['lightStatus/laser'].value[goodIPM],
#        #xray = f['lightStatus/xray'].value[goodIPM]
#        
#        x =f['epix/droplet_X'].value[goodIPM],
#        y = f['epix/droplet_Y'].value[goodIPM],
#        ADU = f['epix/droplet_adu'].value[goodIPM],
#        Npix = f['epix/droplet_npix'].value[goodIPM],
#        #ROI_sum = f['epix/ROI_sum'].value[goodIPM]
#        #ROI_comx = f['epix/ROI_com'].value[goodIPM][:, 0]
#        #ROI_comy = f['epix/ROI_com'].value[goodIPM][:, 1]
#        ipm = f['ipm3/sum'].value[goodIPM],
#        jitter = f['epics/lxt_ttc'].value[goodIPM],
#        laser = f['lightStatus/laser'].value[goodIPM],
#        xray = f['lightStatus/xray'].value[goodIPM]
#                 )
#    for key, name in ScanMotors.items():
#        d[key] = f[name].value[goodIPM]
#    return d
#def get_data(run, ScanMotors={'delay': 'epics/lxt_ttc'}, fpath=None):
#def get_data(run, ScanMotors={'samrot': 'scan/samrot'}, fpath=None):

#def get_data(run, ScanMotors={}, fpath=None):

#def get_rixs(fname, motor_cen, motor_window, y_edges, elPix=300, ScanMotor='ebeam/L3Energy'):
#	""" convenience function combinding get_data and bin_rixs to return RIXS spectrum"""
#	if type(fname) == str:
#		DATA = get_data(fname, ScanMotor=ScanMotor)
#	else:
#		DATA = combine(*[get_data(filename, scanMotor=scanMotor) for filename in fname])
#	y = select_time_ADU(DATA, motor_cen, motor_window)[1]
#	return bin_rixs(y, y_edges, elPix)
