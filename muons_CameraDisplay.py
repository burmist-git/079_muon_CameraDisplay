#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
import gc
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.coordinates import SkyCoord, AltAz, angular_separation
import astropy.units as u
from scipy.stats import binned_statistic
from scipy.signal import lombscargle
import h5py
from astropy.io import fits
from astropy.table import Table
from tables import open_file
from astropy.table import join, vstack
from astropy.stats import sigma_clip
from ctapipe.io import read_table 
from ctapipe.instrument import SubarrayDescription
from matplotlib.colors import LogNorm
import math
import yaml
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from scipy.stats import skew
from scipy.stats import kurtosis
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import copy

from astropy.table import join, vstack
from ctapipe.io import read_table 

from ctapipe.instrument import SubarrayDescription
from matplotlib.colors import LogNorm

import math

from ctapipe.visualization import CameraDisplay


def print_conf_to_canvas(conf, fig, y_pos = 1.0, y_step=0.1):
    """Print configuration to figure (not used)"""


    figure=fig
    plt.axis('off')
    for key, values in conf.items():
        plt.text(0, y_pos, f"{key}: {values}", fontsize=12, va='top')
        y_pos -= y_step

    return figure

def analyze(conf, subarr):
    
    #SST-1M DigiCam (corsika_theta_20.0_az_180.0_run10.simtel.gz)
    #geom = subarr.tel[4].camera.geometry
    #optics = subarr.tel[4].optics
    
    #Mace (corsika_theta_20.0_az_180.0_run10.simtel.gz)
    #geom = subarr.tel[1].camera.geometry
    #optics = subarr.tel[1].optics

    if conf['camera_type'] == 'LSTCam':
        #LST LSTCam
        geom = subarr.tel[1].camera.geometry
        optics = subarr.tel[1].optics
    elif conf['camera_type'] == 'NectarCam':
        #MST NectarCam
        geom = subarr.tel[100].camera.geometry
        optics = subarr.tel[100].optics
    elif conf['camera_type'] == 'FlashCam':
        #MST FlashCam
        geom = subarr.tel[5].camera.geometry
        optics = subarr.tel[5].optics
    elif conf['camera_type'] == 'SST':
        #SST CHEC
        geom = subarr.tel[60].camera.geometry
        optics = subarr.tel[60].optics    
    
    #geom = subarr.tel[50].camera.geometry
    
    #geom = subarr.tel[8].camera.geometry
    #def prod5_mst_flashcam(subarray_prod5_paranal):
    #    return subarray_prod5_paranal.tel[5]
    #def prod5_mst_nectarcam(subarray_prod5_paranal):
    #    return subarray_prod5_paranal.tel[100]
    #def prod5_lst(subarray_prod5_paranal):
    #    return subarray_prod5_paranal.tel[1]
    #def prod5_sst(subarray_prod5_paranal):
    #    return subarray_prod5_paranal.tel[60]
    
    
    disp = CameraDisplay(geom)


    h5file=open_file(conf['file'], "a")
    
    #h5file=open_file("data/run_000/dl1_muon_ctapipe_run000_dev.h5", "a")
    #h5file=open_file("data/test/muon-_0deg_0deg_run000002___cta-prod6-2147m-Paranal-mst-nc-dark-ref-degraded-0.8.h5", "a")
    #h5file=open_file("/home/burmist/Downloads/muon-_0deg_0deg_run000001___cta-prod6-2147m-Paranal-mst-nc-dark-ref-degraded-0.8.simtel.zst", "a")
    #h5file=open_file("./data/data_muon_verticle_no_magnetic_field/muon-_0deg_0deg_run000001___cta-prod6-2147m-Paranal-mst-nc-dark-ref-degraded-0.8.h5", "a")


    #print(h5file.root.dl1.event.telescope.images.tel_001[:])
    #print(h5file.root.dl1.event.telescope.parameters.tel_001[:])
    #h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][0]
    #h5file.root.simulation.event.subarray.shower[:]['true_core_y'][0]
    print(h5file.root.simulation.event.subarray.shower[:]['true_energy'][0])
    
    rr=np.sqrt(h5file.root.simulation.event.subarray.shower[:]['true_core_x'] ** 2 + h5file.root.simulation.event.subarray.shower[:]['true_core_y'] ** 2)
    ee = h5file.root.simulation.event.subarray.shower[:]['true_energy'][:]

    rr_reco=np.sqrt(h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_impact_x'] ** 2 + h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_impact_x'] ** 2)
    muonefficiency_optical_efficiency=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_optical_efficiency']

    #reco_muon_info=h5file.root.dl1.event.telescope.muon.tel_001[:]
    
    
    rr_reco_clean=rr_reco[~np.isnan(rr_reco)]
    muonefficiency_optical_efficiency_clean=muonefficiency_optical_efficiency[~np.isnan(muonefficiency_optical_efficiency)]

    rr=np.sqrt(h5file.root.simulation.event.subarray.shower[:]['true_core_x'] ** 2 + h5file.root.simulation.event.subarray.shower[:]['true_core_y'] ** 2)
    
    hillas_intensity=h5file.root.dl1.event.telescope.parameters.tel_001[:]['hillas_intensity'][:]
    morphology_n_pixels=h5file.root.dl1.event.telescope.parameters.tel_001[:]['morphology_n_pixels'][:]

    muonring_radius=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonring_radius'][:]
    muonring_width=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_width'][:]
    muonparameters_containment=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_containment'][:]


    muonring_center_fov_lon_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonring_center_fov_lon'][:]
    muonring_center_fov_lat_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonring_center_fov_lat'][:]
    muonparameters_ring_intensity_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_ring_intensity'][:]
    muonparameters_n_pixels_in_ring_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_n_pixels_in_ring'][:]
    muonparameters_radial_std_dev_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_radial_std_dev'][:]
    muonparameters_skewness_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_skewness'][:]
    muonparameters_excess_kurtosis_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_excess_kurtosis'][:]
    muonefficiency_width_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_width'][:]
    muonring_radius_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonring_radius'][:]
    muonparameters_mean_intensity_outside_ring_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonparameters_mean_intensity_outside_ring'][:]
    muonefficiency_impact_x_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_impact_x'][:]
    muonefficiency_impact_y_all=h5file.root.dl1.event.telescope.muon.tel_001[:]['muonefficiency_impact_y'][:]

    true_core_x_all=h5file.root.simulation.event.subarray.shower[:]['true_core_x'][:] 
    true_core_y_all=h5file.root.simulation.event.subarray.shower[:]['true_core_y'][:]
    
    impact_resolution_x=muonefficiency_impact_x_all + true_core_y_all
    impact_resolution_y=muonefficiency_impact_y_all + true_core_x_all
    
    hillas_intensity_clean=hillas_intensity[~np.isnan(hillas_intensity)]
    morphology_n_pixels_clean=morphology_n_pixels[~np.isnan(morphology_n_pixels)]

    muonring_radius_clean=muonring_radius[~np.isnan(muonring_radius)]
    muonring_width_clean=muonring_width[~np.isnan(muonring_width)]
    muonparameters_containment_clean=muonparameters_containment[~np.isnan(muonparameters_containment)]


    df = pd.DataFrame(
        {
            'optical_efficiency': muonefficiency_optical_efficiency,
            'r_reco': rr_reco,
            'r': rr,
            'muonring_width': muonring_width,
            'muonring_center_fov_lon': muonring_center_fov_lon_all,
            'muonring_center_fov_lat': muonring_center_fov_lat_all,
            'muonparameters_ring_intensity': muonparameters_ring_intensity_all,
            'muonparameters_n_pixels_in_ring': muonparameters_n_pixels_in_ring_all,
            'muonparameters_radial_std_dev': muonparameters_radial_std_dev_all,
            'muonparameters_skewness': muonparameters_skewness_all,
            'muonparameters_excess_kurtosis': muonparameters_excess_kurtosis_all,
            'muonefficiency_width': muonefficiency_width_all,
            'muonring_radius': muonring_radius_all,
            'muonparameters_mean_intensity_outside_ring': muonparameters_mean_intensity_outside_ring_all,
            'muonefficiency_impact_x': muonefficiency_impact_x_all,
            'muonefficiency_impact_y': muonefficiency_impact_y_all,
            'true_core_x': true_core_x_all,
            'true_core_y': true_core_y_all,
            'impact_resolution_x': impact_resolution_x,
            'impact_resolution_y': impact_resolution_y,
        }
    )

    df_clean = df.dropna()
    df_cut = df_clean[df_clean['r_reco'] > 4]


    
    #print(hillas_intensity)
    #print(morphology_n_pixels)

    #print(len(rr))
    #print(len(ee))
    #print(len(hillas_intensity))
    #print(len(morphology_n_pixels))

    #print(len(muonring_radius))
    #print(len(muonparameters_containment))

    #print(np.count_nonzero(~np.isnan(muonring_radius)))
    #print(np.count_nonzero(~np.isnan(muonparameters_containment)))

    #min_pixels", "np.count_nonzero(mask) > 20"],
    #ring_containment", "parameters.containment > 0.5"]

    
    #for i in np.arange(6):
    #    print(i+1," ",rr[i])
        
    #optics
    #np.sqrt(optics.mirror_area/np.pi)

    
    #
    # image
    #

    with PdfPages(str(conf['file'] + str(".pdf"))) as pdf:



        #hillas_intensity_clean=hillas_intensity[~np.isnan(hillas_intensity)]
        #morphology_n_pixels_clean=morphology_n_pixels[~np.isnan(morphology_n_pixels)]
        #muonring_radius_clean=arr[~np.isnan(muonring_radius)]
        #muonparameters_containment_clean=arr[~np.isnan(muonparameters_containment)]
        #rr=np.sqrt(h5file.root.simulation.event.subarray.shower[:]['true_core_x'] ** 2 + h5file.root.simulation.event.subarray.shower[:]['true_core_y'] ** 2)
        #ee = h5file.root.simulation.event.subarray.shower[:]['true_energy'][:]

    
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        plt.tight_layout()
        axes[0][0].hist(
            df['muonring_center_fov_lon'].values, 
            bins=np.linspace(-2.0,2.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonring_center_fov_lon',
        )
        axes[0][0].legend(fontsize=13)
        axes[0][1].hist(
            df['muonring_center_fov_lat'].values, 
            bins=np.linspace(-2.0,2.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonring_center_fov_lat',
        )        
        axes[0][1].legend(fontsize=13)
        #
        axes[1][0].hist(
            df['muonparameters_ring_intensity'].values, 
            bins=np.linspace(0.0,5000.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_ring_intensity',
        )
        axes[1][0].legend(fontsize=13)
        axes[1][1].hist(
            df['muonparameters_n_pixels_in_ring'].values, 
            bins=np.linspace(0.0,400.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_n_pixels_in_ring',
        )        
        axes[1][1].legend(fontsize=13)
        #
        axes[2][0].hist(
            df['muonparameters_radial_std_dev'].values, 
            bins=np.linspace(0.0,1.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_radial_std_dev',
        )
        axes[2][0].legend(fontsize=13)
        axes[2][1].hist(
            df['muonefficiency_width'].values, 
            bins=np.linspace(0.0,0.2,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonefficiency_width',
        )        
        axes[2][1].legend(fontsize=13)
        #
        axes[3][0].hist(
            df['muonring_radius'].values, 
            bins=np.linspace(0.5,1.3,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonring_radius',
        )
        axes[3][0].legend(fontsize=13)
        axes[3][1].hist(
            df['muonparameters_mean_intensity_outside_ring'].values, 
            bins=np.linspace(-0.3,0.3,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_mean_intensity_outside_ring',
        )        
        axes[3][1].legend(fontsize=13)
        #
        axes[4][0].hist(
            df['muonefficiency_impact_x'].values, 
            bins=np.linspace(-11,11,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonefficiency_impact_x',
        )
        axes[4][0].legend(fontsize=13)
        axes[4][1].hist(
            df['muonefficiency_impact_y'].values, 
            bins=np.linspace(-11,11,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonefficiency_impact_y',
        )        
        axes[4][1].legend(fontsize=13)
        pdf.savefig()
        plt.close()




        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        plt.tight_layout()
        axes[0].hist(
            df['impact_resolution_x'].values, 
            bins=np.linspace(-10.0,10.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='impact_resolution_x',
        )
        axes[0].legend(fontsize=13)
        axes[1].hist(
            df['impact_resolution_y'].values, 
            bins=np.linspace(-10.0,10.0,200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='impact_resolution_y',
        )        
        axes[1].legend(fontsize=13)

        pdf.savefig()
        plt.close()


        
        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            ee,
            bins=np.linspace(0.0,
                             0.050,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='energy',
        )
        #plt.ylim(0,200)
        plt.title("true energy", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('true energy', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()


        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            rr,
            bins=np.linspace(0.0,
                             10,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='true impact r',
        )
        plt.hist(
            rr_reco_clean,
            bins=np.linspace(0.0,
                             10,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='reco impact r',
        )
        #plt.ylim(0,200)
        plt.title("true impact r", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('true impact r', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()



        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            muonefficiency_optical_efficiency_clean,
            bins=np.linspace(0.1,
                             0.3,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonefficiency_optical_efficiency',
        )
        #
        plt.hist(
            df_cut['optical_efficiency'].values,
            bins=np.linspace(0.1,
                             0.3,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonefficiency_optical_efficiency cut',
        )
        #plt.ylim(0,200)
        plt.title("optical_efficiency", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('optical_efficiency', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()
        

        #
        plt.figure(figsize=(10, 10))
        plt.hist2d(
            df_clean['optical_efficiency'].values,
            df_clean['r_reco'].values,           
            bins=[
                np.linspace( 0.1, 0.3, 100), 
                np.linspace( 0.0, 10.0, 100),
            ],
            cmap='plasma'
        )
        #plt.legend(fontsize=15)
        plt.xlabel('optical_efficiency', fontsize=15)
        plt.ylabel('r_reco', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()
        


        
        
        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            hillas_intensity_clean,
            bins=np.linspace(0.0,
                             5000,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='hillas_intensity',
        )
        #plt.ylim(0,200)
        plt.title("hillas intensity", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('hillas intensity', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()

        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            morphology_n_pixels_clean,
            bins=np.linspace(0.0,
                             150,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='morphology_n_pixels',
        )
        #plt.ylim(0,200)
        plt.title("morphology n_pixels", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('morphology n_pixels', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()

        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            muonring_radius,
            bins=np.linspace(0.6,
                             1.3,
                             200),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonring_radius',
        )
        #plt.ylim(0,200)
        plt.title("muonring radius", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('muonring radius', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()

        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            muonring_width_clean,
            bins=np.linspace(0.0,
                             0.2,
                             300),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonring width',
        )
        #plt.ylim(0,200)
        plt.title("muonring width, deg", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('muonring width, deg', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()


        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            df['muonparameters_radial_std_dev'].values,
            bins=np.linspace(0.0,
                             1.0,
                             300),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonring width',
        )
        #plt.ylim(0,200)
        plt.title("muonparameters_radial_std_dev", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('muonparameters_radial_std_dev', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()


        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            df['muonparameters_skewness'].values,
            bins=np.linspace(-50.0,
                             50.0,
                             300),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_skewness',
        )
        #plt.ylim(0,200)
        plt.title("muonparameters_skewness", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('muonparameters_skewness', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()

        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            df['muonparameters_excess_kurtosis'].values,
            bins=np.linspace(-1000.0,
                             1000.0,
                             300),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_excess_kurtosis',
        )
        #plt.ylim(0,200)
        plt.title("muonparameters_excess_kurtosis", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('muonparameters_excess_kurtosis', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()


        
        #
        plt.figure(figsize=(10, 10))
        plt.hist2d(
            df_clean['muonring_radius'].values,
            df_clean['muonring_width'].values,           
            bins=[
                np.linspace( 0.8, 1.3, 100), 
                np.linspace( 0.0, 0.1, 100),
            ],
            cmap='plasma',
            label='clean',
        )
        #plt.legend(fontsize=13)
        plt.title("clean", fontsize=15)
        plt.xlabel('muonring width, deg.', fontsize=15)
        plt.ylabel('muonring radius, deg.', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()

        #
        plt.figure(figsize=(10, 10))
        plt.hist2d(
            df['muonring_radius'].values,
            df['muonring_width'].values,           
            bins=[
                np.linspace( 0.8, 1.3, 100), 
                np.linspace( 0.0, 0.1, 100),
            ],
            #vmin=0,
            #vmax=30,
            cmap='plasma',
            label='all',
        )
        plt.title("all", fontsize=15)
        #plt.legend(fontsize=15)
        plt.xlabel('muonring width, deg.', fontsize=15)
        plt.ylabel('muonring radius, deg.', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()



        #
        plt.figure(figsize=(10, 10))
        plt.hist2d(
            df['muonring_radius'].values,
            df['muonparameters_ring_intensity'].values,           
            bins=[
                np.linspace( 0.8, 1.3, 100), 
                np.linspace( 0.0, 3000, 100),
            ],
            #vmin=0,
            #vmax=30,
            cmap='plasma',
            label='all',
        )
        #plt.title("all", fontsize=15)
        #plt.legend(fontsize=15)
        plt.xlabel('muonring_radius, deg.', fontsize=15)
        plt.ylabel('muonparameters_ring_intensity', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()



        #
        plt.figure(figsize=(10, 10))
        plt.hist2d(
            df['muonring_radius'].values,
            df['muonparameters_mean_intensity_outside_ring'].values,           
            bins=[
                np.linspace( 0.8, 1.3, 100), 
                np.linspace( -0.3, 0.3, 100),
            ],
            #vmin=0,
            #vmax=30,
            cmap='plasma',
            label='all',
        )
        #plt.title("all", fontsize=15)
        #plt.legend(fontsize=15)
        plt.xlabel('muonring_radius, deg.', fontsize=15)
        plt.ylabel('muonparameters_mean_intensity_outside_ring', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()



        

        
        #
        plt.figure(figsize=(15, 10))
        plt.hist(
            muonparameters_containment_clean,
            bins=np.linspace(0.0,
                             1.1,
                             100),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='muonparameters_containment_clean',
        )
        #plt.ylim(0,200)
        plt.title("muonparameters_containment_clean", fontsize=20)
        plt.legend(fontsize=15)
        plt.xlabel('muonparameters_containment_clean', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        pdf.savefig()
        plt.close()


        for i_frames in np.arange(conf['nframes']):

            #plt.tight_layout(pad=3.5)
            plt.tight_layout()
            
            ev_id = i_frames*6
                
            fig_tmp=plt.figure(figsize=(15, 10))
            
            for i in np.arange(2):
                for j in np.arange(3):
                    eventstate = {
                        'ev_ID': ev_id + i*3 + j,
                        'r': rr[ev_id + i*3 + j],
                        'e': ee[ev_id + i*3 + j],
                    }
                    y_step = 0.025
                    y_pos  = 1.0 - 3*y_step*(i*3 + j)
                    print_conf_to_canvas(eventstate, fig_tmp, y_pos, y_step)
                    
                    
            pdf.savefig()
            plt.close()
                    
                    
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                    
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image'][ev_id + 0],
                ax=ax[0][0])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image'][ev_id + 1],
                ax=ax[0][1])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image'][ev_id + 2],
                ax=ax[0][2])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image'][ev_id + 3],
                ax=ax[1][0])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image'][ev_id + 4],
                ax=ax[1][1])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image'][ev_id + 5],
                ax=ax[1][2])
            #print(h5file.root.dl1.event.telescope.images.tel_001[:]['event_id'][5])
            
            ax[0][0].set_title("Image", fontsize=20)
            
            pdf.savefig()
            plt.close()
            
            
        
            #
            # true_image
            #
            
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            disp = CameraDisplay(
                geom,
                h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][ev_id + 0],
                ax=ax[0][0])
            disp = CameraDisplay(
                geom,
                h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][ev_id + 1],
                ax=ax[0][1])
            disp = CameraDisplay(
                geom,
                h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][ev_id + 2],
                ax=ax[0][2])
            disp = CameraDisplay(
                geom,
                h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][ev_id + 3],
                ax=ax[1][0])
            disp = CameraDisplay(
                geom,
                h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][ev_id + 4],
                ax=ax[1][1])
            disp = CameraDisplay(
                geom,
                h5file.root.simulation.event.telescope.images.tel_001[:]['true_image'][ev_id + 5],
                ax=ax[1][2])
            #print(h5file.root.dl1.event.telescope.images.tel_001[:]['event_id'][5])
            ax[0][0].set_title("True image", fontsize=20)
            
            pdf.savefig()
            plt.close()
            
            
            #
            # image_mask
            #
            
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image_mask'][ev_id + 0],
                ax=ax[0][0])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image_mask'][ev_id + 1],
                ax=ax[0][1])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image_mask'][ev_id + 2],
                ax=ax[0][2])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image_mask'][ev_id + 3],
                ax=ax[1][0])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image_mask'][ev_id + 4],
                ax=ax[1][1])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['image_mask'][ev_id + 5],
                ax=ax[1][2])
            ax[0][0].set_title("Image mask", fontsize=20)
            
            pdf.savefig()
            plt.close()
            
            
            #
            # peak_time
            #
            
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['peak_time'][ev_id + 0],
                ax=ax[0][0])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['peak_time'][ev_id + 1],
                ax=ax[0][1])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['peak_time'][ev_id + 2],
                ax=ax[0][2])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['peak_time'][ev_id + 3],
                ax=ax[1][0])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['peak_time'][ev_id + 4],
                ax=ax[1][1])
            disp = CameraDisplay(
                geom,
                h5file.root.dl1.event.telescope.images.tel_001[:]['peak_time'][ev_id + 5],
                ax=ax[1][2])
            ax[0][0].set_title("Peak time", fontsize=20)
            
            pdf.savefig()
            plt.close()

        

    h5file.close()

    
def main():
    """Main program"""


    parser = argparse.ArgumentParser(
        description="muons_CameraDisplay"
    )

    # Add arguments
    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Configuration file"
    )

    
    # Parse arguments
    args = parser.parse_args()

    with open(args.conf, 'r') as file:
        conf = yaml.safe_load(file)

        
        
    file_list = list(conf['file'])
    camera_type_list = list(conf['camera_type'])

    subarr=SubarrayDescription.read(
        "dataset://gamma_prod5.simtel.zst",
        focal_length_choice="EQUIVALENT"
    )


    i=0
    for the_file in conf['file']:
        print(the_file)
        print(camera_type_list[i])
        conf['file'] = the_file
        conf['camera_type'] = camera_type_list[i]
        analyze(conf, subarr)
        i += 1


if __name__ == "__main__":
    main()
