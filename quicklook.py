import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as si
from astropy.table import Table
from astropy.io import fits
import numpy as np
import string
from pathlib import Path
import matplotlib.patches as patches


def plot_plate(file_number=21, annotate=False, hexabundle_scale_factor = 30, marker_size=10, alpha=0.1):
    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 8))

    hdu_AAOmega = fits.open(f"/Volumes/OtherFiles/09dec200{file_number}red.fits")
    hdu_Spectre = fits.open(f"/Volumes/OtherFiles/09dec400{file_number}red.fits")

    ax.set_title(f"Filenames: 09dec200{file_number}.fits & 09dec400{file_number}.fits")
    plate = patches.Circle(xy=(0, 0), radius=236*1000, edgecolor='k', facecolor='None', linewidth=3.0)
    ax.add_patch(plate)
    for hexabundle in list(string.ascii_uppercase[:21]):
        if hexabundle in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            hdu = hdu_AAOmega
        else:
            hdu = hdu_Spectre

        #print(f'{hexabundle}')
        data = hdu[0].data
        df = Table(hdu[2].data).to_pandas()
        df['SPAX_ID'] = df['SPAX_ID'].str.strip()

        mask = df['SPAX_ID'] == hexabundle

        flux_values = np.nanmedian(data, axis=1)[mask]

        # xi_1d = np.arange(df.loc[mask, 'FIBPOS_X'].min(), df.loc[mask, 'FIBPOS_X'].max())
        # yi_1d = np.arange(df.loc[mask, 'FIBPOS_Y'].min(), df.loc[mask, 'FIBPOS_Y'].max())

        # xi, yi = np.meshgrid(xi_1d, yi_1d)


        

        # vals = si.griddata((df.loc[mask, 'FIBPOS_X'], df.loc[mask, 'FIBPOS_Y']), values=flux_values, xi=(xi, yi),method='cubic')

        
        #ax.imshow(np.flipud(vals), extent=np.array([xi.min(), xi.max(), yi.min(), yi.max()]), cmap='plasma')
        centre_x = df.loc[mask, 'FIBPOS_X'].mean()
        centre_y = df.loc[mask, 'FIBPOS_Y'].mean()
        hexabundle_xvals = (df.loc[mask, 'FIBPOS_X'] - centre_x) * hexabundle_scale_factor# + centre_x
        hexabundle_yvals = (df.loc[mask, 'FIBPOS_Y'] - centre_y) * hexabundle_scale_factor# + centre_y


        angs = df.loc[mask, 'ANGS'].mean() + np.pi
        x_rot = np.cos(angs) * hexabundle_xvals + np.sin(angs) * hexabundle_yvals # Flipped this
        y_rot = -np.sin(angs) * hexabundle_xvals + np.cos(angs) * hexabundle_yvals

        X = x_rot + centre_x
        Y = y_rot + centre_y

        


        xi_1d = np.linspace(X.min(), X.max(), 100)
        yi_1d = np.linspace(Y.min(), Y.max(), 100)
        xi, yi = np.meshgrid(xi_1d, yi_1d)

        vals = si.griddata((X, Y), values=flux_values, xi=(xi, yi),method='cubic')

        ax.imshow(np.flipud(vals), extent=np.array([xi.min(), xi.max(), yi.min(), yi.max()]), cmap='plasma', zorder=9)

        #ax.scatter(X, Y, c=flux_values, cmap='plasma', edgecolors='k', alpha=alpha, s=marker_size)

        if annotate:
            ax.annotate(f"{hexabundle}", xy=(X.mean(), Y.mean()), xytext=(-15, -15), textcoords='offset points', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), ha='left', va='top', zorder=10)
        
        #fib_num_mask = df.loc[mask, 'FIBNUM'].isin([1, 3, 11, 25,45])
        #ax.scatter(X[fib_num_mask], Y[fib_num_mask], c='r', edgecolors='k', alpha=0.8)

        #Plot a line of angle angs
        length = 50000
        ax.plot([centre_x, centre_x + length * np.cos(angs)], [centre_y, centre_y + length * np.sin(angs)], c='k', linewidth=3.0, alpha=0.5, zorder=1)

        #Plot a line of angle PORIENT
        porient = np.radians(df.loc[mask, 'PORIENT'].mean())
        ax.plot([centre_x, centre_x + length * np.cos(porient)], [centre_y, centre_y + length * np.sin(porient)], c='cyan', linewidth=3.0, alpha=1, zorder=1)
    
    return fig, ax
        #ax.set_title(f'{hexabundle}')



    



# ###All hexabundles close up
def plot_hexas():
    fig, axs = plt.subplots(ncols=5, nrows=6, figsize=(15, 10), constrained_layout=True)

    hdu_AAOmega = fits.open("/Volumes/OtherFiles/09dec20021red.fits")
    hdu_Spectre = fits.open("/Volumes/OtherFiles/09dec40021red.fits")

    for ax, hexabundle in zip(axs.ravel(), list(string.ascii_uppercase[:21])):
        if hexabundle in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            hdu = hdu_AAOmega
        else:
            hdu = hdu_Spectre

        print(f'{hexabundle}')
        data = hdu[0].data
        df = Table(hdu[2].data).to_pandas()
        df['SPAX_ID'] = df['SPAX_ID'].str.strip()

        mask = df['SPAX_ID'] == hexabundle

        xi_1d = np.arange(df.loc[mask, 'FIBPOS_X'].min(), df.loc[mask, 'FIBPOS_X'].max())
        yi_1d = np.arange(df.loc[mask, 'FIBPOS_Y'].min(), df.loc[mask, 'FIBPOS_Y'].max())

        xi, yi = np.meshgrid(xi_1d, yi_1d)


        flux_values = np.nanmedian(data, axis=1)[mask]

        vals = si.griddata((df.loc[mask, 'FIBPOS_X'], df.loc[mask, 'FIBPOS_Y']), values=flux_values, xi=(xi, yi),method='cubic')


        ax.imshow(np.flipud(vals), extent=np.array([xi.min(), xi.max(), yi.min(), yi.max()]), cmap='plasma')
        ax.scatter(df.loc[mask, 'FIBPOS_X'], df.loc[mask, 'FIBPOS_Y'], c=flux_values, cmap='plasma', edgecolors='k', alpha=0.8)
        fib_num_mask = df.loc[mask, 'FIBNUM'].isin([1, 3, 11, 25,45])
        ax.scatter(df.loc[mask, 'FIBPOS_X'].values[fib_num_mask], df.loc[mask, 'FIBPOS_Y'].values[fib_num_mask], c='r', edgecolors='k', alpha=0.8)
        ax.set_title(f'{hexabundle}')



    for ax in axs.ravel():
        ax.axis('off')

# plt.show()


def plot_rotation(hexabundle):

    #How one hexabundle c
    rotation_dict = {100:'29', 150:'27', 200:'25', 250:'23', 300:'21', 350:'24', 400:"26", 450:'28', 500:'30'}

    fig, axs = plt.subplots(ncols=len(rotation_dict), figsize=(20, 5), constrained_layout=True)
    for ax, rotation_value in zip(axs.ravel(), rotation_dict):

        if hexabundle in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            file = Path(f"/Volumes/OtherFiles/09dec200{rotation_dict[rotation_value]}red.fits")
        else:
            file = Path(f"/Volumes/OtherFiles/09dec400{rotation_dict[rotation_value]}red.fits")

        print(f'{rotation_value}')
        if not file.exists():
            print("No data!")
            continue

        hdu = fits.open(file)
        data = hdu[0].data
        df = Table(hdu[2].data).to_pandas()
        df['SPAX_ID'] = df['SPAX_ID'].str.strip()

        mask = df['SPAX_ID'] == hexabundle

        xi_1d = np.arange(df.loc[mask, 'FIBPOS_X'].min(), df.loc[mask, 'FIBPOS_X'].max())
        yi_1d = np.arange(df.loc[mask, 'FIBPOS_Y'].min(), df.loc[mask, 'FIBPOS_Y'].max())

        xi, yi = np.meshgrid(xi_1d, yi_1d)


        flux_values = np.nanmedian(data, axis=1)[mask]

        vals = si.griddata((df.loc[mask, 'FIBPOS_X'], df.loc[mask, 'FIBPOS_Y']), values=flux_values, xi=(xi, yi),method='cubic')


        ax.imshow(np.flipud(vals), extent=np.array([xi.min(), xi.max(), yi.min(), yi.max()]), cmap='plasma')
        ax.scatter(df.loc[mask, 'FIBPOS_X'], df.loc[mask, 'FIBPOS_Y'], c=flux_values, cmap='plasma', edgecolors='k', alpha=0.8)

        fib_num_mask = df.loc[mask, 'FIBNUM'].isin([1, 3, 11, 25,45])
        ax.scatter(df.loc[mask, 'FIBPOS_X'].values[fib_num_mask], df.loc[mask, 'FIBPOS_Y'].values[fib_num_mask], c='r', edgecolors='k', alpha=0.8)
        ax.set_title(f'{rotation_value}')

    for ax in axs.ravel():
        ax.axis('off')
    fig.suptitle(f"Hexabundle {hexabundle}")
    plt.show()
    # fig.savefig('')
    # plt.show()



if __name__ == "__main__":

    from tqdm import tqdm
    rotation_dict = {100:'29', 150:'27', 200:'25', 250:'23', 300:'21', 350:'24', 400:"26", 450:'28', 500:'30'}
    for rotation in tqdm(rotation_dict):
        fname = Path(f"/Volumes/OtherFiles/09dec200{rotation_dict[rotation]}red.fits")
        if fname.exists():
            fig,ax = plot_plate(file_number=rotation_dict[rotation], annotate=True, hexabundle_scale_factor=40, marker_size=15, alpha=0.0)
            ax.set_title(f"{rotation} millidegrees")
            fig.savefig(f"RotationPlots/{rotation}.pdf", bbox_inches='tight')
            plt.close()
        else:
            print(f"No Data for {rotation}")
            continue
    pass
    #plot_rotation('S')
    #plot_plate()
    #plot_hexas()
