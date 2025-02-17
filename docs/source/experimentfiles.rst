Creating experiment files
=========================

#. if you need a mask file for your data, make a mask by typing 

    .. code-block:: bash

        makemask -dir path/to/experiment/directory -s ##scan##number#
    
    This will open up the mask GUI. Save the created mask and note down the file path ###/###/#####.edf

#. if you do not have an experiment file, make an experiment setup file by typing 

    .. code-block:: bash

        makesetup 
        
    this will open up a template experimental file, edit with your experimental information. this will include
        a. edfmaskfile   = path to the mask you just created.
        b. local_output_path
        c. local_data_path
        d. beam_centre
        e. detector_distance
        f. ensure your process_outputs are correct - explained lower down 

#. Process outputs explained

    a. **'full_reciprocal_map':**  calculates a full reciprocal space map combining all scans listed into a single volume. Use this option for scan such as crystal truncation rod scans, fractional order rod scans, or  in-plane HK scans.
    b. **'pyfai_qmap':** calculates 2d q_parallel Vs q_perpendicular plots using pyFAI. Use this options for GIWAXS measurements either with a static detector or a moving detector.
    c. **'pyfai_ivsq':** calculates 1d Intensity Vs Q using pyFAI. Use this options for GIWAXS measurements either with a static detector or a moving detector.
    d. **'pyfai_exitangles':** calculates a 2d map of horizontal exit angle Vs vertical exit angle

    the **map_per_image** option will set whether to combine all scans into a single HKL volume (map_per_image=False) , or whether to analyse each image individually into its own HKL volume (map_per_image=True)

    .. collapse:: Deprecated options which have been removed

        #.  **'pyfai_2dqmap_IvsQ'** - use parallel multiprocessing to calculate both 2d Qpara Vs Qperp map, as well as 1d  Intensity Vs Q integration - both using pyFAI package

        #. **'large_moving_det'**  utilise MultiGeometry option in pyFAI for scan with a moving detector and a large number of images (~1000s), outputs: I, Q, two theta, caked image,Q_para Vs Q_perp

        #. **'curved_projection_2D'**   (use 'large_moving_det' option instead) this projects a series of detector images into a single 2D image, treating the images as if there were all from a curved detector. This will give a projected 2d image. This should only be used when a detector has been moved on the diffractometer arm during a scan, and the images need to be combined together. NOTE: this will not work on a continuous scans with ~1000s of images - for these scan types use the 'large_moving_det' option. 
        
        #. **'pyfai_1D'** - ( use 'pyfai_2dqmap_IvsQ' option instead)Does an azimuthal integration on an image using PONI and MASK settings described in corresponding files. This can handle a short series of images (~50) for individual integrations. If used in combination with 'curved_projection_2D', this will simply integrate the large projected image, and not the series of small images. 

        #. **'qperp_qpara_map'** - ( use 'pyfai_2dqmap_IvsQ' option instead) projects GIWAXS image into q_para,q_perp plot.  NOTE: Similar to 'curved_projection_2D', this will not work with 1000's of images - for these scans use the 'large_moving_det' option. 
#. save this exp_setup.py file noting down the local_data_path
#. Usually the default calc_setup.py file that contains the calculation settings will be suitable,however if a bespoke calc_setup.py is needed, copy over the calc_setup.py file in the fast_rsm/CLI/i07  folder and edit accordingly. contact beamline staff or i07 data analysis scientist for guidance on this.