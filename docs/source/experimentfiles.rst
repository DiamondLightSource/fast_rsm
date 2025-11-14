Creating experiment files
=========================



To create a new experiment setup file type the following command after loading the fast_rsm module

    .. code-block:: bash

        makesetup 
    
this will open up a template experimental file, edit with your experimental information. Use the information below to create your exp_setup.py file. The sections are organised as follows:
    
    * :ref:`info_for_all_scans` - These are all of the values which always need to be defined in your experimental setup file 
    * :ref:`optional_for_all` - These are the optional values which can be set for all scan types, but will default to a preset values if not included in the experimental setup file
    * :ref:`optional_GIWAXS` - These are the optional values which can be set for GIWAXS type scans, and will default to preset values if not included in the experimental setup file
    * :ref:`optional_singlecrystal`- These are the optional values which can be set for HKL reciprocal space type scans,and will default to preset values if not included in the experimental setup file

Usually the default calc_setup.py file that contains the calculation settings will be suitable,however if a bespoke calc_setup.py is needed, copy over the calc_setup.py file in the fast_rsm/CLI/i07  folder and edit accordingly. contact beamline staff or i07 data analysis scientist for guidance on this.

If you need a mask file for your data, make a mask by using the makemask command from the terminal line. This requires two inputs:
    
    * -dir =the path to where the experiment data i stored
    * -s =the number of the scan you want to make the mask for

    e.g.

    .. code-block:: bash

        makemask -dir /dls/i07/data/2025/si36456-5/sample1 -s 535612
    
    This will open up the mask GUI. Save the created mask and note down the full file path to the .edf file.

.. _info_for_all_scans:

Information required for all scan types
--------------------------------------------

.. confval:: setup

        How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'

.. confval:: experimental_hutch

        which experimental hutch was used 1= experimental hutch 1, 2=experimental hutch 2



.. confval:: local_data_path

        Set this to the directory path where your files are saved, note you will need to include any subdirectories in this path e.g 
        
        .. code-block:: bash
            
            /dls/i07/data/2025/si36456-5/sample1

.. confval:: local_output_path

        Set this to the path where you want the output from the data processing to be saved e.g.

        .. code-block:: bash
            
            /dls/i07/data/2025/si36456-5/processing/sample1 
    
.. confval:: beam_centre
    
    The beam centre, as can be read out from GDA, in pixel_x, pixel_y.

.. confval:: detector_distance

    The distance between the sample and the detector (or, if using the DCD, the distance between the receiving slit and the detector). Units of meters.

.. confval:: process_outputs 

    define what outputs you would like from the processing in a list e.g. ['pyfai_ivsq']. The options available are:

    * **'full_reciprocal_map':**  calculates a full reciprocal space map combining all scans listed into a single volume. Use this option for scan such as crystal truncation rod scans, fractional order rod scans, or  in-plane HK scans.
    * **'pyfai_qmap':** calculates 2d q_parallel Vs q_perpendicular plots using pyFAI. Use this options for GIWAXS measurements either with a static detector or a moving detector.
    * **'pyfai_ivsq':** calculates 1d Intensity Vs Q using pyFAI. Use this options for GIWAXS measurements either with a static detector or a moving detector.
    * **'pyfai_exitangles':** calculates a 2d map of horizontal exit angle Vs vertical exit angle

.. confval:: map_per_image

    this option will set whether to combine all scans into a single output:

    - map_per_image=False -> gives a single output file (either HKL volume or mapped GIWAXS data)
    
    - map_per_image=True -> analyses each image individually creating multiple output files (either HKL volumes or mapped GIWAXS)



.. _optional_for_all:

Optional settings for all scan types
--------------------------------------


Masking
.........

.. confval:: edfmaskfile

    add path to the .edf mask file. This file can be created with pyFAI gui accessed via the 'makemask' command


.. confval:: specific_pixels

    If you have a small number of hot pixels to mask, specify them one at a time in a list. Which should look like:
    specific_pixels = [(pixel_x1, pixel_y1), (pixel_x2, pixel_y2)] e.g. to mask pixel (233, 83) and pixel (234, 83), where pixel coordinates are (x, y) use  

    .. code-block:: python

        specific_pixels=[(233, 234),(83, 83)]

.. confval:: mask_regions

    to set masking on specific areas, define regions with start_x,  stop_x, start_y, stop_y, and then set the mask_regions to a list of these regions. e.g. 

    .. code-block:: python
            
        mask_1 = (0, 75, 0, 194)
        mask_2 = (425, 485, 0, 194)
        mask_regions = [mask_1,mask_2]

.. confval:: min_intensity

    set a minimum intensitys, where pixels with and intensity below this value are masked


If there are also some corrupted images within a scan you can define which images to skip in which scans using the following setings:

.. confval::  skipscans

    set which scans have images to skip - giving the scan numbers in a list e.g. 
    
    .. code-block:: python

        skipscans=[535612,535643]   

.. confval:: skipimages

    specify which images within the skipscans need to be skipped, given as a list of lists e.g. 
    
    .. code-block:: python
        
        skipimages=[[1,3,5],[4,7,9]]


Detector Positioning System settings
...........................................

.. confval:: using_dps

    When using the Detector Positioning System add this variable as True to the experimental setup file

    If you are using DPS you will also need to add the dps central pixels with the three variables below. 
    The DPS central pixel locations are not typically recorded in the nexus file. This should be the central pixel for the undeflected beam with units of meters. 


    .. confval:: dpsx_central_pixel
    .. confval:: dpsy_central_pixel
    .. confval:: dpsz_central_pixel



.. _optional_GIWAXS:    

Optional settings for GIWAXS analysis
--------------------------------------


There will always be a .hdf5 file created. You can set the option for exporting additonal files with the savetiffs and savedats options below

.. confval:: savetiffs

    if you want to export '2d qpara Vs qperp maps' to extra .tiff images set to True

.. confval:: savedats

    if you want to export '1d I Vs Q data' to extra .dat files set to True

When using slits positioned between the detector and the sample, the distances need to be included

.. confval:: slitvertratio

    if using vertical slits between sample and detector set to slit-detector-distance/sample-detector-distance  e.g. 0.55/0.89

.. confval:: slithorratio

    if using horizontal slits between sample and detector set to slit-detector-distance/sample-detector-distance  e.g. 0.55/0.89

.. confval:: alphacritical

    define the critical edge of the sample material in degrees, when using DCD and wanting to account for extra exit angles

To set the mapping limits for 1D  integrations use the following settings

.. confval:: radialrange

    specify the range of two theta to calculate the 1D data for e.g. (0,60)

To manually set the step size you can use one of two options:


    .. confval:: radialstepval 

        specify the size of step in degrees between each binned data point in two theta

    OR

    .. confval:: ivqbins

        specify directly the number of bins for I Vs Q profile - which will mean radialstepval will have no effect

Note: the number of bins will be automatically calculated using a step size of 0.01 degrees if no step settings are given.


.. _optional_singlecrystal:

Optional settings for full reciprocal space maps
-------------------------------------------------

below are a list of optional settings for full reciprocal space maps, including CTRs and in-plane omega scans

.. confval:: volume_start
    
    The starting limits of the HKL volume given as  [h_start, k_start, l_start] e.g. [1,1,0.2]

.. confval:: volume_stop
    
    The end limits of the HKL volume given as  [h_stop, k_stop, l_stop]  e.g. [1,1,3.5]

To define the resolution of the HKL map you can specify one of the two values:


    .. confval:: volume_step
        
        The steps between voxels in the HKL volume given as [h_step,k_step,l_step]  e.g. [0,0,0.02]

    OR

    .. confval:: output_file_size

        you can specify an upper limit on the size of the file (in MB) to be output when mapping a full reciprocal space map, this will then calculate the highest resolution for the speficied HKL limits that can still say under the file size limit

.. confval:: save_vtk

    Choose if you want a .vtk volume saved as well the hdf5, which can be used for loading into paraview

.. confval:: save_npy

    Choose if you want a .npy file saved as well as the hdf5, for manual analysis




.. confval:: load_from_dat

        Only include this if you need to load your data from a .dat file. 


.. confval:: coordinates

    Choose to set the co-ordinate system to use for the full reciprocal space map. Options for coordinates argument are:

        * **'cartesian'** - normal cartesian coords: hkl, Qx Qy Qz, etc.
        * **'cylindricalpolar'** - cylindrical polar with cylinder axis set by the cylinder_axis variable
        * **'sphericalpolarr'** - spherical polar with centre set by the spherical_bragg_vec variable

.. confval:: cylinder_axis

    choose to have a manually set cylinder axis

.. confval:: frame_name

    choose to change frame of reference for mapping from default 'hkl' :

        * **'qxqyqz'** - map data into cartesian Qx,Qy,Qz values, which is useful when lattice of material is none cubic e.g. hexagonal. 

.. confval:: spherical_bragg_vec
    
    define a vector to shift the centre point of mapped volume to a specific bragg peak

