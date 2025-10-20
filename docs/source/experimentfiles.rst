Creating experiment files
=========================

#. if you need a mask file for your data, make a mask by typing 

    .. code-block:: bash

        makemask -dir path/to/experiment/directory -s ##scan##number#
    
    This will open up the mask GUI. Save the created mask and note down the file path ###/###/#####.edf

#. if you do not have an experiment file, make an experiment setup file by typing 

    .. code-block:: bash

        makesetup 
        
    this will open up a template experimental file, edit with your experimental information. Use the information below to create your exp_setup.py file


Information required for all scan types
--------------------------------------------

    .. collapse:: setup

            How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'

    .. collapse:: experimental_hutch

            which experimental hutch was used 1= experimental hutch 1, 2=experimental hutch 2
    
    .. collapse:: local_data_path

            Set this to the directory path where your files are saved, note you will need to include any subdirectories in this path e.g 
            
            .. code-block:: bash
                
                /dls/i07/data/2025/si36456-5/sample1

    .. collapse:: local_output_path

            Set this to the path where you want the output from the data processing to be saved e.g.

            .. code-block:: bash
                
                /dls/i07/data/2025/si36456-5/processing/sample1 
        
    .. collapse:: beam_centre
        
        The beam centre, as can be read out from GDA, in pixel_x, pixel_y.

    .. collapse:: detector_distance

        The distance between the sample and the detector (or, if using the DCD, the distance between the receiving slit and the detector). Units of meters.

Optional settings for all scan types
--------------------------------------



Masking
.........

    .. collapse:: edfmaskfile

        add path to the .edf mask file. This file can be created with pyFAI gui accessed via the 'makemask' command


    .. collapse:: specific_pixels

        If you have a small number of hot pixels to mask, specify them one at a time in a list. Which should look like:
        specific_pixels = [(pixel_x1, pixel_y1), (pixel_x2, pixel_y2)] e.g. to mask pixel (233, 83) and pixel
         (234, 83), where pixel coordinates are (x, y) use  [(233, 234),(83, 83)]
    
    .. collapse:: mask_regions

        to set masking on specific areas, define regions with start_x,  stop_x, start_y, stop_y, and then set the mask_regions to a list of these regions.
        e.g. 
        .. code-block:: python
               
            mask_1 = (0, 75, 0, 194)
            mask_2 = (425, 485, 0, 194)
            mask_regions = [mask_1,mask_2]
    
    .. collapse:: min_intensity

        set a minimum intensitys, where pixels with and intensity below this value are masked

    
    If there are also some corrupted images within a scan you can define which images to skip in which scans using the following setings:

    .. collapse::  skipscans

        set which scans have images to skip - giving the scan numbers in a list e.g. [535612,535643]   

    .. collapse:: skipimages

        specify which images within the skipscans need to be skipped, given as a list of lists e.g. [[1,3,5],[4,7,9]]


Detector Positioning System settings
...........................................

    .. collapse:: using_dps

        Are you using the Detector Positioning System? set to True or False
    
        if you are using DPS then you will need to add the dps central pixels
        # The DPS central pixel locations are not typically recorded in the nexus file.
        # NOTE THAT THIS SHOULD BE THE CENTRAL PIXEL FOR THE UNDEFLECTED BEAM.
        # UNITS OF METERS, PLEASE (everything is S.I., except energy in eV).

        .. code-block:: python

            dpsx_central_pixel = 0
            dpsy_central_pixel = 0
            dpsz_central_pixel = 0
    


    
Optional settings for GIWAXS analysis
--------------------------------------

    .. collapse:: slitvertratio

        if using vertical slits between sample and detector set to slit-detector-distance/sample-detector-distance  e.g. 0.55/0.89

    .. collapse:: slithorratio

        if using horizontal slits between sample and detector set to slit-detector-distance/sample-detector-distance  e.g. 0.55/0.89

    .. collapse:: alphacritical

        define the critical edge of the sample material in degrees, when using DCD and wanting to account for extra exit angles

To set the mapping limits for 1D  integrations use the following settings

    .. collapse:: radialrange

        specify the range of two theta to calculate the 1D data for e.g. (0,60)

 To manually set the step size you can use one of two options:


    .. collapse:: radialstepval 

        specify the size of step in degrees between each binned data point in two theta

    OR

    .. collapse:: ivqbins
        :class: blue-title

        specify directly the number of bins for I Vs Q profile - which will mean radialstepval will have no effect

Note: the number of bins will be automatically calculated using a step size of 0.01 degrees if no step settings are given.



Optional settings for single crystal XRD
----------------------------------------------

below are a list of optional setting for single crystal XRD type scans including CTRs and in-plane omega scans

.. collapse:: volume_start
    
    The starting limits of the HKL volume given as  [h_start, k_start, l_start] e.g. [1,1,0.2]

.. collapse:: volume_stop
    
    The end limits of the HKL volume given as  [h_stop, k_stop, l_stop]  e.g. [1,1,3.5]

To define the resolution of the HKL map you can specify one of the two values:


    .. collapse:: volume_step
        
        The steps between voxels in the HKL volume given as [h_step,k_step,l_step]  e.g. [0,0,0.02]

    OR

    .. collapse:: output_file_size

        you can specify an upper limit on the size of the file (in MB) to be output when mapping a full reciprocal space map, this will then calculate the highest resolution for the speficied HKL limits that can still say under the file size limit

.. collapse:: save_vtk

    Choose if you want a .vtk volume saved as well the hdf5, which can be used for loading into paraview

.. collapse:: save_npy

    Choose if you want a .npy file saved as well as the hdf5, for manual analysis




        # choose map co-ordinates for special mappings e.g. polar co-ordinates, if commented out defaults to co-ordinates='cartesian'
        # coordinates='sphericalpolar'

        # choose central point to calculate spherical polars around - if commented out defaults to [0,0,0]
        # spherical_bragg_vec=[1.35,1.42,0.96] #519910 , 519528


.. collapse:: loadload_from_dat_from_dat

        Only include this if you need to load your data from a .dat file. 




    .. code-block:: python






        # *******calculating azimuthal integrations from single images to give I Vs Q plots

        # *********calculating qpara Vs qperp maps,
        



        



        # ============OUTPUTS==========
        # define what outputs you would like form the processing here, choose from:
        # 'full_reciprocal_map' = calculates a full reciprocal space map combining all
        #                           scans listed into a single volume
        #
        # 'pyfai_qmap' = calculates 2d q_parallel Vs q_perpendicular plots using pyFAI
        #
        # 'pyfai_ivsq' = calculates 1d Intensity Vs Q using pyFAI
        #
        # 'pyfai_exitangles' - calculates a map of vertical exit angle Vs horizontal exit angle

        # 'pyfai_ivsq'  , 'pyfai_qmap','pyfai_exitangles' ,'full_reciprocal_map'
        process_outputs = ['pyfai_qmap']


        # Set this to True if you would like each image to be mapped independently.
        # If this is False, all images in all scans will be combined into one large
        # reciprocal space map.
        map_per_image = False

        # There will always be a .hdf5 file created. You can set the option for exporting additonal files with the savetiffs and savedats options below
        # if you want to export '2d qpara Vs qperp maps' to extra .tiff images set
        # savetiffs to True
        savetiffs = False

        # if you want to export '1d I Vs Q data' to extra .dat files set savedats
        # to True
        savedats = False

        # # Options for coordinates argument are:
        # #     'cartesian'   (normal cartesian coords: hkl, Qx Qy Qz, etc.)
        # #     'cylindricalpolar'       (cylindrical polar with cylinder axis set by the
        # #                        cylinder_axis variable)
        # #     'sphericalpolarr     (spherical polar with centre set by the
        # #                        spherical_bragg_vec variable)
        coordinates='cartesian' 
        cylinder_axis=False
        #use this vector to shift centre point of mapped volume to a specific bragg peak
        spherical_bragg_vec=[0,0,0] 

        # The scan numbers of the scans that you want to use to produce this reciprocal
        # space map.    




        

#. Process outputs explained

    a. **'full_reciprocal_map':**  calculates a full reciprocal space map combining all scans listed into a single volume. Use this option for scan such as crystal truncation rod scans, fractional order rod scans, or  in-plane HK scans.
    b. **'pyfai_qmap':** calculates 2d q_parallel Vs q_perpendicular plots using pyFAI. Use this options for GIWAXS measurements either with a static detector or a moving detector.
    c. **'pyfai_ivsq':** calculates 1d Intensity Vs Q using pyFAI. Use this options for GIWAXS measurements either with a static detector or a moving detector.
    d. **'pyfai_exitangles':** calculates a 2d map of horizontal exit angle Vs vertical exit angle

    the **map_per_image** option will set whether to combine all scans into a single output:
        - map_per_image=False --> gives a single output file (either HKL volume or mapped GIWAXS data)
        - map_per_image=True --> analyses each image individually creating multiple output files (either HKL volumes or mapped GIWAXS)

    .. collapse:: Deprecated options which have been removed

        #.  **'pyfai_2dqmap_IvsQ'** - use parallel multiprocessing to calculate both 2d Qpara Vs Qperp map, as well as 1d  Intensity Vs Q integration - both using pyFAI package

        #. **'large_moving_det'**  utilise MultiGeometry option in pyFAI for scan with a moving detector and a large number of images (~1000s), outputs: I, Q, two theta, caked image,Q_para Vs Q_perp

        #. **'curved_projection_2D'**   (use 'large_moving_det' option instead) this projects a series of detector images into a single 2D image, treating the images as if there were all from a curved detector. This will give a projected 2d image. This should only be used when a detector has been moved on the diffractometer arm during a scan, and the images need to be combined together. NOTE: this will not work on a continuous scans with ~1000s of images - for these scan types use the 'large_moving_det' option. 
        
        #. **'pyfai_1D'** - ( use 'pyfai_2dqmap_IvsQ' option instead)Does an azimuthal integration on an image using PONI and MASK settings described in corresponding files. This can handle a short series of images (~50) for individual integrations. If used in combination with 'curved_projection_2D', this will simply integrate the large projected image, and not the series of small images. 

        #. **'qperp_qpara_map'** - ( use 'pyfai_2dqmap_IvsQ' option instead) projects GIWAXS image into q_para,q_perp plot.  NOTE: Similar to 'curved_projection_2D', this will not work with 1000's of images - for these scans use the 'large_moving_det' option. 
#. save this exp_setup.py file noting down the local_data_path
#. Usually the default calc_setup.py file that contains the calculation settings will be suitable,however if a bespoke calc_setup.py is needed, copy over the calc_setup.py file in the fast_rsm/CLI/i07  folder and edit accordingly. contact beamline staff or i07 data analysis scientist for guidance on this.