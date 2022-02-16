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
    parser.add_argument("file_number", type=int)
    parser.add_argument("--config-file")
    parser.add_argument("--outfile")
    parser.add_argument("-s", "--sigma-clip", action='store_true')
    args = parser.parse_args()

    obs_number = args.file_number

    config_filename = 'hector_display_config_SWAPS.yaml'
    if args.config_file is not None:
        config_filename = args.config_file

    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_filename} does not exist!")

    
    # Load some options
    # If the sigma clip option is True in the config file, turn it on
    # If it's False in the config file, check for the command line option
    sigma_clip = config['sigma_clip']
    if not sigma_clip:
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


    # Get the swaps dictionary
    # If it's a dictionary, use it as normal
    # If it's a filename pointing to a yaml file, load that
    swaps = config['swaps_dictionary']
    if type(swaps) == str:
        with open(swaps, 'r') as g:
            swaps = yaml.safe_load(g)['swaps_dictionary']
    
    # Do some checks on it
    # Check we have the right number of key/value pairs
    assert len(swaps) == 21, f"Seem to have an incorrect number of elements in the swap list: {len(swaps)} instead of 21"
    # Check that no hexabundles are repeated
    assert len(list(swaps.keys())) == len(set(swaps.keys())), "We seem to have a repeated value in the LHS of the swaps dictionary"
    assert len(list(swaps.values())) == len(set(swaps.values())), "We seem to have a repeated value in the RHS of the swaps dictionary"
    # And check that the values are equal to the set of letters from A to U
    assert set(swaps.keys()) == set(string.ascii_uppercase[:21]), "The LHS of the swaps list is incorrect- are there any missing hexabundle letters?"
    assert set(swaps.values()) == set(string.ascii_uppercase[:21]), "The RHS of the swaps list is incorrect- are there any missing hexabundle letters?"

    # Get the fibre tables and find the tramlines:
    object_fibtab_A, object_guidetab_A, object_spec_A, spec_id_alive_A = utils.get_alive_fibres(flat_file_AAOmega, object_file_AAOmega, sigma_clip=sigma_clip)
    object_fibtab_H, object_guidetab_H, object_spec_H, spec_id_alive_H = utils.get_alive_fibres(flat_file_Hector, object_file_Hector, sigma_clip=sigma_clip)

    # Plot the data
    print("---> Plotting...")
    print("--->")

    scale_factor = 18

    fig = plt.figure(figsize=(10,10))
    
    fig.suptitle(f"Hector raw data with hexabundle swaps: {config['file_prefix']} frame {obs_number}",fontsize=15)

    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')

    ax.add_patch(Circle((0,0), 264/2*2000, facecolor="#cccccc", edgecolor='#000000', zorder=-1))

    for Probe in list(string.ascii_uppercase[:21]):

        if Probe in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            object_fibtab, object_guidetab, object_spec, spec_id_alive = object_fibtab_A, object_guidetab_A, object_spec_A, spec_id_alive_A
        else:
            object_fibtab, object_guidetab, object_spec, spec_id_alive = object_fibtab_H, object_guidetab_H, object_spec_H, spec_id_alive_H


        swapped_probe = swaps[Probe]

        if swapped_probe in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            swapped_object_fibtab, swapped_object_guidetab, swapped_object_spec, swapped_spec_id_alive = object_fibtab_A, object_guidetab_A, object_spec_A, spec_id_alive_A
        else:
            swapped_object_fibtab, swapped_object_guidetab, swapped_object_spec, swapped_spec_id_alive = object_fibtab_H, object_guidetab_H, object_spec_H, spec_id_alive_H

        print(f"Probe {Probe} is now in position {swapped_probe}")
        #print(f"Probe = {Probe}, swapped with {swapped_probe}")
        mask = (object_fibtab.field('TYPE')=="P") & (object_fibtab.field('SPAX_ID')==Probe)
        swapped_mask = (swapped_object_fibtab.field('TYPE')=="P") & (swapped_object_fibtab.field('SPAX_ID')==swapped_probe)
        
        Probe_data = object_spec[spec_id_alive[mask]]

        #mask = np.logical_and(object_fibtab.field('TYPE')=="P",\object_fibtab['SPAX_ID']==Probe)

        mean_x = np.mean(swapped_object_fibtab.field('MAGX')[swapped_mask])
        mean_y = np.mean(swapped_object_fibtab.field('MAGY')[swapped_mask])

        x = 1 * (object_fibtab.field('FIB_PX')[mask])
        y = 1 * (object_fibtab.field('FIB_PY')[mask])

        original_probe_angle = np.unique(object_fibtab.field('ANGS')[mask])
        angle = np.unique(swapped_object_fibtab.field('ANGS')[swapped_mask])

        assert len(angle) == 1, 'Must only have one angle per probe'

        rotation_angle_original = original_probe_angle - np.pi/2
        rotation_angle = angle - np.pi/2

        x_rotated = -1 * (np.cos(rotation_angle_original) * x - np.sin(rotation_angle_original) * y)
        y_rotated = -1 * (np.sin(rotation_angle_original) * x + np.cos(rotation_angle_original) * y)

        length = scale_factor * 1000
        line_hexabundle_tail = [(mean_x, mean_y), (mean_x + length * np.sin(rotation_angle), mean_y - length * np.cos(rotation_angle))]

        #Plot the hexabundle tail
        ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=2, zorder=1, alpha=0.5)

        # Add the collection of circles
        ax.add_collection(utils.display_ifu(x_rotated, y_rotated, mean_x, mean_y, scale_factor, Probe_data))
        ax.axis([-140000*2, 140000*2, -140000*2, 140000*2])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.text(mean_x, mean_y - scale_factor*750*2, f"Probe {Probe} in position {swapped_probe}",               verticalalignment="bottom", horizontalalignment='center')



    for probe_number, hexabundle_x, hexabundle_y, angle in zip(
        object_guidetab['PROBENUM'], object_guidetab['CENX'], object_guidetab['CENY'], object_guidetab['ANGS']):

        rotation_angle = angle - np.pi/2
        ax.add_patch(Circle((hexabundle_x,hexabundle_y), scale_factor*250, edgecolor='#009900', facecolor='none'))
        ax.text(hexabundle_x, hexabundle_y, f"G{probe_number - 21}",
                verticalalignment='center', horizontalalignment='center')
        line_hexabundle_tail = [(hexabundle_x, hexabundle_y), (hexabundle_x + length * np.sin(rotation_angle), hexabundle_y - length * np.cos(rotation_angle))]
        ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=1, zorder=1)

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
