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

import hector_display_utils as utils


if __name__ == "__main__":
    

    import yaml
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("CCD")
    parser.add_argument("file_number", type=int, help='The object frame number you want to display')
    parser.add_argument("--config-file", help="A .yaml file which contains parameters and filenames for the code. See hector_display_config.yaml for an example")
    parser.add_argument("--outfile", help='Filename to save the plot to. If not given, display the plot instead')
    parser.add_argument("-s", "--sigma-clip", action='store_true', help='Turn on sigma clipping. Can also be set in the config file')
    args = parser.parse_args()

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

    plot_fibre_traces = True

    if not object_file_Hector.exists():
        raise FileNotFoundError(f"The Hector file seems to not exist: {object_file_Hector} not found")
    if not object_file_AAOmega.exists():
        raise FileNotFoundError(f"The AAomega file seems to not exist: {object_file_AAOmega} not found")

    # Get the fibre tables and find the tramlines:
    object_fibtab_A, object_guidetab_A, object_spec_A, spec_id_alive_A = utils.get_alive_fibres(flat_file_AAOmega, object_file_AAOmega, sigma_clip=sigma_clip, IFU="unknown", log=True, pix_waveband=100, pix_start="unknown", plot_fibre_trace = plot_fibre_traces)
    object_fibtab_H, object_guidetab_H, object_spec_H, spec_id_alive_H = utils.get_alive_fibres(flat_file_Hector, object_file_Hector, sigma_clip=sigma_clip, IFU="unknown", log=True, pix_waveband=100, pix_start="unknown", plot_fibre_trace = plot_fibre_traces)

    # Plot the data
    print("---> Plotting...")
    print("--->")

    scale_factor = 18
    hexabundle_tail_length = scale_factor * 1000

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

        
        line_hexabundle_tail = [(mean_x, mean_y), (mean_x + hexabundle_tail_length * np.sin(rotation_angle), mean_y - hexabundle_tail_length * np.cos(rotation_angle))]
        ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=2, zorder=1, alpha=0.5)


        ax.add_collection(utils.display_ifu(x_rotated, y_rotated, mean_x, mean_y, scale_factor, Probe_data))
        ax.axis([-140000*2, 140000*2, -140000*2, 140000*2])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.text(mean_x, mean_y - scale_factor*750*2, "Probe " + str(Probe),\
                verticalalignment="bottom", horizontalalignment='center')


    # Add the guides
    ax = utils.display_guides(ax, object_guidetab, scale_factor=scale_factor, tail_length=hexabundle_tail_length)

    # And add some N/E arrows
    ax = utils.add_NE_arrows(ax)

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
