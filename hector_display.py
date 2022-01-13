"""
This module is a modified version of sami_display intended for a quick view of Hector data.

This module has quite a lot of fudges and magic numbers that work fine for the Hector Commissioning data
but may not always work for other data sets.

Edited by Jesse van de Sande Dec-2021
Edited by Sam Vaughan Jan 2021

"""
import numpy as np
import scipy as sp

import astropy.io.fits as pf
from astropy.io import fits

import string
import itertools

# Circular patch.
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox

import string

#import pdb
#import PyQt5

#########################################################################################

def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    
    """
    #
    # "peakdetect"
    #
    #   Determines peaks from data. Translation of the MATLAB code "peakdet.m"
    #   and taken from https://gist.github.com/sixtenbe/1178136
    #
    #   Called by "raw"
    #
    """
    
    i = 10000
    x = np.linspace(0, 3.5 * np.pi, i)
    y = (0.3*np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x) + 0.06 * np.random.randn(i))
    
    # Converted from/based on a MATLAB script at:
    # http://billauer.co.il/peakdet.html
    
    # function for detecting local maximas and minmias in a signal.
    # Discovers peaks by searching for values which are surrounded by lower
    # or larger values for maximas and minimas respectively
    
    # keyword arguments:
    # y_axis........A list containg the signal over which to find peaks
    # x_axis........(optional) A x-axis whose values correspond to the y_axis list
    #               and is used in the return to specify the postion of the peaks. If
    #               omitted an index of the y_axis is used. (default: None)
    # lookahead.....(optional) distance to look ahead from a peak candidate to
    #               determine if it is the actual peak (default: 200)
    #               '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    # delta.........(optional) this specifies a minimum difference between a peak and
    #               the following points, before a peak may be considered a peak. Useful
    #               to hinder the function from picking up false peaks towards to end of
    #               the signal. To work well delta should be set to delta >= RMSnoise *
    #               5. (default: 0)
    # delta.........function causes a 20% decrease in speed, when omitted. Correctly used
    #               it can double the speed of the function
    # return........two lists [max_peaks, min_peaks] containing the positive and negative
    #               peaks respectively. Each cell of the lists contains a tupple of:
    #               (position, peak_value) to get the average peak value do:
    #               np.mean(max_peaks, 0)[1] on the results to unpack one of the lists
    #               into x, y coordinates do: x, y = zip(*tab)
    #
    
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
    
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
    
    return [max_peaks, min_peaks]

def _datacheck_peakdetect(x_axis, y_axis):
    """Used as part of "peakdetect" """
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    
def display_ifu(x_coords, y_coords, xcen, ycen, scaling, values, zorder=10):
    bundle_patches = []
    for x1,y1 in zip(x_coords, y_coords):
        circle = Circle((x1*scaling+xcen,y1*scaling+ycen), 52.5*scaling)
        bundle_patches.append(circle)
    pcol = PatchCollection(bundle_patches, cmap=plt.get_cmap('afmhot'), zorder=zorder)
    pcol.set_array(values)
    pcol.set_edgecolors('none')
    return pcol

class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.from_bounds(0, 0, 1, 1),
                                self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])



def plot_guide_rotations(df):

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df.x, df.y, c='r', zorder=10)
    length = 10


    for index, row in df.iterrows():

        if not 'Sky' in str(row.ID):
            angle_degrees = (row.angs + np.pi) * 180/np.pi

            if angle_degrees>360:
                angle_degrees -= 360

            line_angle_eq_0 = [(row.x, row.y), (row.x + 7.5, row.y)]
            line_hexabundle_tail = [(row.x, row.y), (row.x + length * np.cos(row.angs + np.pi), row.y + length * np.sin(row.angs + np.pi))]
            ax.annotate(f'{row.Hexabundle}', xy=(row.x, row.y), xytext=(-20, -20), textcoords='offset points', size=12, fontweight='bold')
            ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=2)
            ax.plot(*zip(*line_angle_eq_0), c='k', linestyle='dashed', linewidth=0.5)
            am = AngleAnnotation((row.x, row.y), line_angle_eq_0[1], line_hexabundle_tail[1], ax=ax, size=75, text=rf"$\alpha={angle_degrees:.1f}$")


    arrow_x_centre = 0.8
    arrow_y_centre = 0.95
    arrow_length = 0.1
    ax.arrow(x=arrow_x_centre, y=arrow_y_centre, dx=0.0, dy=-1 * arrow_length, transform=ax.transAxes, width=0.005, facecolor='k')
    ax.arrow(x=arrow_x_centre, y=arrow_y_centre, dx=arrow_length, dy=0.0, transform=ax.transAxes, width=0.005, facecolor='k')

    #N arrow
    ax.annotate('N', xy=(arrow_x_centre, arrow_y_centre), xytext=(arrow_x_centre, arrow_y_centre - 0.03 - arrow_length), xycoords=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='center')
    ax.annotate('E', xy=(arrow_x_centre, arrow_y_centre), xytext=(arrow_x_centre + arrow_length + 0.03, arrow_y_centre), xycoords=ax.transAxes,
            fontsize=16, fontweight='bold', va='center', ha='left')
    return fig, ax



def get_alive_fibres(flat_file, object_file, IFU="unknown", sigma_clip=True, log=True, pix_waveband=100,     pix_start="unknown", figfile=None, plot_fibre_trace = False):


    """
    # "raw"
    #
    #   Takes in a raw flat field and a raw object frame. Performs a cut on the flat
    #   field along the centre of the CCD to get fibre row positions.
    #   Collapses +/- 50 wavelength pix on the object frame at those positions
    #   and plots them.
    #
    #   Function Example:
    #
    #       sami.display.raw("02sep20045.fits","02sep20053.fits",Probe_to_fit=2,
    #                   sigma_clip=True)
    #
    #   Input Parameters:
    #
    #       flat_file.......File name string of the raw flat frame to find tramlines
    #                       on (e.g. "02sep20045.fits").
    #
    #       object_file.....File name string of the object frame wanting to be
    #                       displayed (e.g. "02sep20048.fits").
    #
    #       IFU.............Integer value to only display that IFU
    #
    #       sigma_clip......Switch to turn sigma clip on and off. If it is on the
    #                       code will run ~20s slower for a pix_waveband of 100. If
    #                       turned off there is a chance that cosmic rays/bad pixels
    #                       will dominate the image stretch and 2D Gauss fits. It is
    #                       strongly advised to turn this on when dealing with the
    #                       Blue CCD as there are many bad pixels. In the Red CCD you
    #                       can normally get away with leaving it off for the sake of
    #                       saving time.
    #
    #       log.............Switch to select display in log or linear (default is log)
    #
    #       pix_waveband....Number of pixels in wavelength direction to bin over,
    #                       centered at on the column of the spatial cut. 100pix is
    #                       enough to get flux contrast for a good fit/image display.
    #
    #       pix_start.......This input is for times where the spatial cut finds 819
    #                       peaks but doesn't find only fibres (e.g. 817 fibres and
    #                       2 cosmic rays). This will be visible in the display
    #                       output and if such a case happens, input the pixel
    #                       location where the previous spatial cut was performed and
    #                       the code will search for better place where 819 fibres
    #                       are present. Keep doing this until 819 are found, and if
    #                       none are found then something is wrong with the flat
    #                       frame and use another.
    #
    """
    


    print("---> START")
    print("--->")
    print(("---> Object frame: "+str(object_file)))
    print("--->")
    
    # Import flat field frame
    flat = pf.open(flat_file)
    flat_data = flat['Primary'].data

    # Import object frame
    object_frame = pf.open(object_file)
    object_data = object_frame['Primary'].data
    object_fibtab = object_frame['MORE.FIBRES_IFU'].data
    object_guidetab = object_frame['MORE.FIBRES_GUIDE'].data
    object_guidetab = object_guidetab[object_guidetab['TYPE']=='G']
    
    # Range to find spatial cut
    if pix_start != "unknown":
        cut_loc_start = np.float(pix_start+5)/np.float(2048)+200
        cut_locs = np.linspace(cut_loc_start,0.75,201)
    else:
        cut_locs = np.linspace(0.25,0.75,201)
    
    if flat['Primary'].header['INSTRUME'] == "AAOMEGA-HECTOR":
        # Fibres B 120 and 175 are just fine, no idea why masked
        object_fibtab.field('TYPE')[120-1] = "P"
        object_fibtab.field('TYPE')[175-1] = "P"
    
        #fibre 637 is broken for all purposes
        object_fibtab.field('TYPE')[637-1] = "U"

    if flat['Primary'].header['INSTRUME'] == "SPECTOR":
        #there is one broken fibre, can't identify so fudge one
        object_fibtab.field('TYPE')[151-1] = "U"
        
        
    #take a record of all alive fibres
    naliveP = np.squeeze(np.where( (object_fibtab.field('TYPE')=="P")))
    naliveS = np.squeeze(np.where( (object_fibtab.field('TYPE')=="S")))
    ndeadU = np.squeeze(np.where( (object_fibtab.field('TYPE')=="U")))
    ndeadN = np.squeeze(np.where( (object_fibtab.field('TYPE')=="N")))
    nalive = np.squeeze(np.where((object_fibtab.field('TYPE')=="P") |
                                 (object_fibtab.field('TYPE')=="S")))
    
    if ndeadU is None:
        ndeadU = 0
    
    print('alive fibres:   ',np.size(naliveP),' + ',np.size(naliveS),' = ',np.size(naliveP)+np.size(naliveS))
    print('offline fibres:   ',np.size(ndeadU),' + ', np.size(ndeadN),' = ',np.size(ndeadU)+np.size(ndeadN))
    
    print("HBundle where deadU",(object_fibtab.field('SPAX_ID')[ndeadU]))
    print("Fibre where deadU",(object_fibtab.field('SPEC_ID')[ndeadU]))
    print("HBundle where deadN",(object_fibtab.field('SPAX_ID')[ndeadN]))
    print("Fibre where deadN",(object_fibtab.field('SPEC_ID')[ndeadN]))
    
    if flat['Primary'].header['INSTRUME'] == "AAOMEGA-HECTOR":
        fitlim = 0.1
    if flat['Primary'].header['INSTRUME'] == "SPECTOR":
            fitlim = 0.075
            
    print("---> Finding suitable cut along spatial dimension...")
    # Check each spatial slice until number of alive fibres (peaks) have been found
    for cut_loc in cut_locs:
        # perform cut along spatial direction
        flat_cut = flat_data[:,int(np.shape(flat_data)[1]*cut_loc)]
        flat_cut_leveled = flat_cut - fitlim*np.max(flat_cut)
        flat_cut_leveled[flat_cut_leveled < 0] = 0.
        # find peaks (fibres)
        peaks = peakdetect(flat_cut_leveled, lookahead = 2)
        Npeaks = np.shape(peaks[0])[0]
        #stop when the number matches the expecte number of #fibres
        if Npeaks == np.size(naliveP)+np.size(naliveS): 
            break
        else:
            continue
    
    print(("---> Spatial cut at pixel number: ",int(cut_loc*2048)))
    print(("---> Number of waveband pixels: ",pix_waveband))
    print(("---> Number of fibres found: ",np.shape(peaks[0])[0]))
    print("--->")    
    
    if plot_fibre_trace:
        tmp_peaks = np.array(peaks[0])    
        fig, ax = plt.subplots(figsize=(30,10))
        ax.plot(np.arange(len(flat_cut_leveled)),flat_cut_leveled,'r')
        c = ax.scatter(tmp_peaks[:,0], tmp_peaks[:,1], s=10, c='b')
        for ai in np.arange(len(tmp_peaks[:,0])):
            ax.annotate(str(ai+1),(tmp_peaks[ai,0], tmp_peaks[ai,1]),c='k',fontsize=10)
        plt.show()
    
    print("--->")
        
    # If the right amount of fibres can't be found then exit script. 
    if Npeaks != len(naliveP)+len(naliveS):
        raise ValueError("---> Can't find right amount of fibres. Check [1] Flat Field is correct [2] Flat Field is supplied as the first variable in the function. If 1+2 are ok then use the 'pix_start' variable and set it at least 10 pix beyond the previous value (see terminal for value)")
    
    # Location of fibre peaks for linear tramline
    tram_loc=[]
    for i in np.arange(np.shape(peaks[0])[0]):
        tram_loc.append(peaks[0][i][0])
    
    
    # Perform cut along spatial direction at same position as cut_loc
    object_cut = object_data[:,int(np.shape(object_data)[1]*cut_loc)-
                             int(pix_waveband/2):int(np.shape(object_data)[1]*cut_loc)+int(pix_waveband/2)]
    
    # "Sigma clip" to get set bad pixels as row median value
    if sigma_clip == True:
        print("---> Performing 'Sigma-clip'... (~20s)")
        for i in np.arange(np.shape(object_cut)[0]):
            for j in np.arange(np.shape(object_cut)[1]):
                med = np.median(object_cut[i,:])
                err = np.absolute((object_cut[i,j]-med)/med)
                if err > 0.25:
                    object_cut[i,j] = med
        print("--->")
    
    # Collapse spectral dimension
    object_cut_sum = np.nansum(object_cut,axis=1)
    
    # Extract intensities at fibre location and log
    object_spec = object_cut_sum[tram_loc]
    
    #provide a tracer array to guide tram_loc to fibre id
    spec_id_alive = np.arange(object_fibtab.field('SPEC_ID')[-1])*0 #+ np.NaN
    spec_id_alive[nalive] = np.arange(Npeaks)

    return object_fibtab, object_guidetab, object_spec, spec_id_alive



if __name__ == "__main__":
    

    import yaml
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("CCD")
    parser.add_argument("file_number", type=int)
    parser.add_argument("--config-file")
    parser.add_argument("--outfile")
    parser.add_argument("-s", "--sigma-clip", action='store_true')
    args = parser.parse_args()

    # ccd = args.CCD
    #flat_number = args.flat_number
    obs_number = args.file_number

    config_filename = 'hector_display_config.yaml'
    if args.config_file is not None:
        config_filename = args.config_file

    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_filename} does not exist!")

    
    # Load some options
    sigma_clip = config['sigma_clip']
    if (args.sigma_clip is not None):
        sigma_clip = args.sigma_clip

    if config['red_or_blue'] == 'blue':
        hector_ccd = 3
        aaomega_ccd = 1
    elif config['red_or_blue'] == 'red':
        hector_ccd = 4
        aaomega_ccd = 2
    else:
        raise NameError(f"The red_or_blue value must be either 'red' or blue': currently {config['red_or_blue']}")

    flat_file_Hector = Path(config['flat_file_Hector'])
    flat_file_AAOmega = Path(config['flat_file_AAOmega'])
    object_file_Hector = Path(config['data_dir']) / f"ccd_{hector_ccd}" / f"{config['file_prefix']}{hector_ccd}{obs_number:04}.fits"
    object_file_AAOmega = Path(config['data_dir']) / f"ccd_{aaomega_ccd}" / f"{config['file_prefix']}{aaomega_ccd}{obs_number:04}.fits"

    if not object_file_Hector.exists():
        raise FileNotFoundError(f"The Hector file seems to not exist: {object_file_Hector} not found")
    if not object_file_AAOmega.exists():
        raise FileNotFoundError(f"The AAomega file seems to not exist: {object_file_AAOmega} not found")

    # Get the fibre tables and find the tramlines:
    object_fibtab_A, object_guidetab_A, object_spec_A, spec_id_alive_A = get_alive_fibres(flat_file_AAOmega, object_file_AAOmega, sigma_clip=sigma_clip)
    object_fibtab_H, object_guidetab_H, object_spec_H, spec_id_alive_H = get_alive_fibres(flat_file_Hector, object_file_Hector, sigma_clip=sigma_clip)

    # Plot the data
    print("---> Plotting...")
    print("--->")

    scale_factor = 18

    fig = plt.figure(figsize=(10,10))
    
    fig.suptitle(f"Hector raw data: {config['file_prefix']} frame {obs_number}",fontsize=15)

    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')

    ax.add_patch(Circle((0,0), 264/2*2000, facecolor="#cccccc", edgecolor='#000000', zorder=-1))


    for Probe in list(string.ascii_uppercase[:21]):

        if Probe in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            object_fibtab, object_guidetab, object_spec, spec_id_alive = object_fibtab_A, object_guidetab_A, object_spec_A, spec_id_alive_A
        else:
            object_fibtab, object_guidetab, object_spec, spec_id_alive = object_fibtab_H, object_guidetab_H, object_spec_H, spec_id_alive_H

        mask = (object_fibtab.field('TYPE')=="P") & (object_fibtab.field('SPAX_ID')==Probe)
        Probe_data = object_spec[spec_id_alive[mask]]

        #mask = np.logical_and(object_fibtab.field('TYPE')=="P",\object_fibtab['SPAX_ID']==Probe)

        mean_x = np.mean(object_fibtab.field('MAGX')[mask])
        mean_y = np.mean(object_fibtab.field('MAGY')[mask])

        x = 1 * (object_fibtab.field('FIB_PX')[mask])
        y = 1 * (object_fibtab.field('FIB_PY')[mask])

        angle = np.unique(object_fibtab.field('ANGS')[mask])
        assert len(angle) == 1, 'Must only have one angle per probe'

        rotation_angle = angle - np.pi/2

        x_rotated = -1 * (np.cos(rotation_angle) * x - np.sin(rotation_angle) * y)
        y_rotated = -1 * (np.sin(rotation_angle) * x + np.cos(rotation_angle) * y)

        length = scale_factor * 1000
        line_hexabundle_tail = [(mean_x, mean_y), (mean_x + length * np.sin(rotation_angle), mean_y - length * np.cos(rotation_angle))]
        ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=2, zorder=1, alpha=0.5)


        ax.add_collection(display_ifu(x_rotated, y_rotated, mean_x, mean_y, scale_factor, Probe_data))
        ax.axis([-140000*2, 140000*2, -140000*2, 140000*2])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.text(mean_x, mean_y - scale_factor*750*2, "Probe " + str(Probe),\
                verticalalignment="bottom", horizontalalignment='center')



    for probe_number, hexabundle_x, hexabundle_y, angle in zip(
        object_guidetab['PROBENUM'], object_guidetab['CENX'], object_guidetab['CENY'], object_guidetab['ANGS']):

        rotation_angle = angle - np.pi/2
        ax.add_patch(Circle((hexabundle_x,hexabundle_y), scale_factor*250, edgecolor='#009900', facecolor='none'))
        ax.text(hexabundle_x, hexabundle_y, f"G{probe_number - 21}",
                verticalalignment='center', horizontalalignment='center')
        line_hexabundle_tail = [(hexabundle_x, hexabundle_y), (hexabundle_x + length * np.sin(rotation_angle), hexabundle_y - length * np.cos(rotation_angle))]
        ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=1, zorder=1)


    ax.arrow(200000,200000,0,30000, facecolor="#aa0000", edgecolor='#aa0000', width=100)
    ax.text(201000,116000*2, 'North', verticalalignment="bottom", horizontalalignment='left')

    ax.arrow(200000,200000,30000,0, facecolor="#aa0000", edgecolor='#aa0000', width=0)
    ax.text(116000*2,201000, 'East', verticalalignment="bottom", horizontalalignment='left')

    plt.tight_layout()
    # if figfile:
    #     plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)
    # #fig.show()

    print("---> END")
    
    if args.outfile is not None:
        fig.savefig(Path(args.outfile), bbox_inches='tight')
    else:
        plt.show()
    #plt.close()
