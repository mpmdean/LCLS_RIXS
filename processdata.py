import numpy as np
import pandas as pd
import h5py
import os
from collections import OrderedDict
import matplotlib.pyplot as plt


"""
Funtions for analyzing XPP small data
"""

ADU_per_photon = 195.
min_ADU = 170.
I0_key = 'ipm3/sum'

# Throw away shots less than 5% of mean
ipm2_thresh = 1.165 * 0.05
ipm3_thresh = 1.177 * 0.05

# Throw away values outside of
corrected_delay_limits = (-0.6, 0.6)

#folder = '/reg/d/psdm/xpp/xpplp7515/hdf5/smalldata'
folder = '/Users/markdean/Documents/Iridates/SACLA_time_resolved/LCLS_October_2017/compressed_data'
def get_filename(run):
    return os.path.join(folder, 'xpplp7515_Run{:03d}.h5'.format(run))

ipm_threshold  = -np.inf
eV_per_pix = 0.023 # from earlier data


def get_Motors(f, Motors=OrderedDict()):
    """ Get dictionary defining which motors to load.
    This is of form.
    {<key in dictionary> : <path in h5 file>}
    """
    # Load droplets information
    Motors.update(OrderedDict(
        x ='epix/droplet_X',
        y = 'epix/droplet_Y',
        ADU = 'epix/droplet_adu',
        Npix = 'epix/droplet_npix',
        ipm2 = 'ipm2/sum',
        ipm3 = 'ipm3/sum',
        jitter = 'epics/lxt_ttc',
        laser = 'lightStatus/laser',
        xray = 'lightStatus/xray',
        delay = 'epics/lxt_vitara'
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

    # Load scan motor
    if len(f['scan'].keys()) >=3:
        first_key = 'scan/' + list(f['scan'].keys())[0]
        Motors[first_key] = first_key

    # add phase cavity
    Motors.update({'delay_correction': 'phase_cav/fit_time_2'})
    # is 'diodeU': 'diodeU/channels' needed??

    return Motors

###########################################
###########################################

class Scan:
    """Container for returning data from h5 files suitable for RIXS.
    This is initalized with a list of runs and falls back on a list of filenames.
    Multiple runs will be contatenated.
    It also allows the user to define a dictionary of motors, which alias the (often cumersome)
    names in the h5 file. These can be loaded into either an ordered dictionary or a xarray object. """
    def __init__(self, RunList):
        try:
            self.FileNameList = [get_filename(run) for run in RunList]
            self.h5files = [h5py.File(filename, 'r') for filename in self.FileNameList]
        except OSError:
            self.FileNameList = RunList
            self.h5files = [h5py.File(filename, 'r') for filename in self.FileNameList]

        self.Motors = get_Motors(self.h5files[0])

    def get_key(self, key):
        """ return the value atribute correspoinding to key"""
        shape = self.h5files[0][key].shape
        if len(shape) == 1:
            return np.hstack(h5file[key].value for h5file in self.h5files)
        else:
            return np.vstack(h5file[key].value for h5file in self.h5files)

    def __getitem__(self, key):
        """ return key where key is a h5 key or its alias defined in self.Motors"""
        try:
            value = self.get_key(self.Motors[key])
        except KeyError:
            value = self.get_key(key)
        except KeyError:
            print("key {} not found".format(key))

        return value

    def get_d(self, ExtraMotors=OrderedDict()):
        """ Return a dictionary will all the motors loaded and combined"""
        d = OrderedDict()
        Motors = self.Motors.copy()
        Motors.update(ExtraMotors)

        for key, name in Motors.items():
            try:
                d[key] = self[key]
            except KeyError:
                print("key {} not found!".format(key))
        try:
            d['corrected_delay'] = self['lxt']*1.*10**12 + self['delay_correction']
        except KeyError:
            print('Failed to create corrected_delay')
            pass
        return d

    def get_filenames(self):
        """ Return string describing filenames"""
        return  "##### Scan Class #####\n##### H5 files are #####\n" + "\n".join([h5file.filename for h5file in self.h5files])

    def get_description(self):
        description = self.get_filenames()
        """Summarize name and keys in files"""
        description += "\n\n#####Contents of 1st file#####\n"
        for key in self.h5files[0].keys():
            try:
                description += "##### " + key + ' #####\n'
                list_of_sub_keys = []
                self.h5files[0][key].visit(list_of_sub_keys.append)
                description += "\t".join(list_of_sub_keys)
                description += "\n"
            except AttributeError:
                description += "##### " + key + ' unexplorable #####\n'
                pass
        return description

    def __repr__(self):
        return self.get_filenames()

    def __str__(self):
        """Printing description"""
        return self.get_description()


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
    Return a selected version of d. Defined by a tolerance range around motorval.
    '''
    goodshots = np.logical_and(d[motorkey]>motorval-tolerance, d[motorkey]<=motorval+tolerance)
    d_select = OrderedDict()
    for key in d.keys():
        d_select[key] = d[key][goodshots]
    return d_select

def select_min_max(d, motorkey, minvalue, maxvalue):
    '''
    Return a selected version of d. Deined as being in between min and max.
    '''
    values = d[motorkey]
    # assign nans to inf in order to avoid warning with < / >
    values[np.isnan(values)] = np.inf
    goodshots = np.logical_and(values>minvalue, values<=maxvalue)
    #print("Min max returning {} pc ".format(100*sum(goodshots) / goodshots.size ))
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

def bin_edges_centers(minvalue, maxvalue, binsize):
    """Make bin edges and centers for use in histogram
    minvalue=-5, maxvalue=5 binsize=1
    Returns
    centers -4, -3, ....4
    edges -4.5, -3.5, ...., 4.5
    ie.. the rounding of the bins edges is such that all bins are fully populated.
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
    edges = binsize * (0.5 + np.arange(minvalue//binsize, maxvalue//binsize))
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

def get_RIXS(run_list, ADU_per_photon=190, minADU=170, maxADU=1000, minx=140, maxx=240, miny=100, maxy=380, binsize=1, elPix=0, laser_select=None, xray_select=None, slope=0):
    """Fast get of RIXS data"""
    S = Scan(run_list)
    d = S.get_d()

    # Remove bad shots
    ipm2 = np.nanmean(S['ipm2'])
    ipm3 = np.nanmean(S['ipm3'])
    d = select_min_max(d, 'ipm2', ipm2_thresh, np.infty)
    d = select_min_max(d, 'ipm3', ipm3_thresh, np.infty)
    d = select_min_max(d, 'corrected_delay', *corrected_delay_limits)

    if laser_select != None:
        d = select(d, 'laser', laser_select)
    if xray_select != None:
        d = select(d, 'xray', xray_select)

    d = clean_data(d, selectkey='ADU', minkey=minADU, maxkey=maxADU, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    x, y, M = make_image(d)
    pixels_edges, pixel_centers = bin_edges_centers(miny, maxy, binsize)
    E, I = bin_rixs_slope(d['x'], d['y'], pixels_edges, d['ADU']/ADU_per_photon, elPix = elPix, slope=slope)
    M /= ADU_per_photon
    X, Y = np.meshgrid(x, y)

    RIXS = OrderedDict(S=S, d=d, E=E, I=I, M=M, X=X, Y=Y, run_list=run_list, )
    return RIXS

def getI0(RIXS):
    """ Return I0"""
    return np.nansum(RIXS['d']['ipm2'])

def plotM(ax, RIXS, vmaxp=98, **kwargs):
    """Plot the image"""
    art = ax.pcolormesh(RIXS['X'], RIXS['Y'], RIXS['M'], vmin=0, vmax=np.percentile(RIXS['M'], vmaxp), **kwargs)
    cb = plt.colorbar(art, ax=ax)
    cb.set_label('Photons')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(RIXS['run_list'])

    return art

def plotRIXS(ax, RIXS, xmin=-0.5, xmax=1.5, norm=1, **kwargs):
    """ Plot the RIXS spectrum"""
    LineArt, *_ = ax.plot(RIXS['E'], RIXS['I'] / norm, '-', alpha=0.5)
    art = ax.errorbar(RIXS['E'], RIXS['I'] / norm, np.sqrt(RIXS['I']) / norm, fmt='o', color=LineArt.get_color(), **kwargs)
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('I')
    ax.set_title(RIXS['run_list'])
    ax.set_xlim([xmin, xmax])

    return art

def save_RIXS(allRIXS, folder='.'):
    """Save two npz arrays with everything apart from the S class."""
    for config in ['on', 'off']:
        save = allRIXS[config].copy()
        save.pop('S')
        filename = "_".join(['{}'.format(run) for run in save['run_list']] + [config])
        filepath = os.path.join(folder, filename)
        print("Save under {}".format(filepath))
        np.savez(filepath, **save)
