Walkthrough 4 - Optional settings applicable to full reciprocal space maps
============================================================================



Resoution information
------------------------------------

You can edit the resolution of your reciprocal space map by setting the following values:

.. confval:: volume_start

    the values to start for each axis, given in the form [hstart,kstart,lstart]

.. confgval:: volume_end

    the values to end for each axis, given in the form [hend, kend, lend]

If the start and stop values are not provided, then the limits are calculated for the whole dataset. 

.. confval:: volume_step

    the step size to use for each axis, given in the form [hstep, kstep, lstep]

note that if volume_step is None then the stepsize is calculated from the output_file_size, which defaults to 100MB. 

.. confval:: output_file_size

    the size limit for the volume file being saved, given in MB. Usually between 100-1000 works well. There is currently a hard limit of 2000. Changing this file size value will have no effect if volume step values have already been provided. 

Alternative dataloading
---------------------------

.. confval:: load_from_dat
    
    Only set this to tru if you need to load your data from a .dat file.


Co-ordinate settings
-----------------------

.. confval:: coordinates

    choose map co-ordinates for special mappings e.g. polar co-ordinates, if commented out defaults to coordinates='cartesian'


.. confval:: spherical_bragg_vec

    choose central point to calculate spherical polars around - if commented out defaults to [0,0,0]

.. confval:: frame_name
    
    choose to change frame of reference for mapping from default ‘hkl’ to 'qxqyqz'.

Optional outputs
------------------

The full reciprocal volumes will be automatically saved as a .hdf5 file, but you can also choose to have extra files created from the volume. 



.. confval:: save_vtk

    Choose if you want a .vtk volume saved as well the hdf5, which can be used for loading into paraview

.. confval:: save_npy

    Choose if you want a .npy file saved as well as the hdf5, for manual analysis


.. tabs::

    .. tab:: Without comments

        .. code-block:: python
            
            # ===================================================================
            # =========Optional settings for full reciprocal space maps
            # =============================================================
            volume_start = None
            volume_stop = None
            volume_step = None
            output_file_size = 50

            save_vtk = False
            save_npy = False

            load_from_dat = True

            coordinates='sphericalpolar'
            spherical_bragg_vec=[1.35,1.42,0.96] 
            frame_name='hkl'

    .. tab:: With comments

        .. code-block:: python
            
            # ===================================================================
            # =========Optional settings for full reciprocal space maps
            # ===================================================================
            #
            # volume_start = [h_start, k_start, l_start]
            # volume_stop = [h_stop, k_stop, l_stop]
            # volume_step = [h_step, k_step, l_step]
            # Leave as None if you don't want to specify them. You can specify whichever
            # you like (e.g. you can specify step and allow start/stop to be auto
            # calculated)
            volume_start = None
            volume_stop = None
            volume_step = None

            # How large would you like your output file to be, in MB? 100MB normally gives
            # very good resolution without sacrificing performance. If you want something
            # higher resolution, feel free, but be aware that the performance of the map and
            # the analysis will start to suffer above around 1GB.
            # Max file size is 2GB (2048MB).
            output_file_size = 50

            # Choose if you want a .vtk volume saved as well the hdf5, which can be used for loading into paraview
            save_vtk = False

            # Choose if you want a .npy file saved as well as the hdf5, for manual analysis
            save_npy = False

            # Only use this if you need to load your data from a .dat file.
            load_from_dat = False
            # choose map co-ordinates for special mappings e.g. polar co-ordinates, if commented out defaults to co-ordinates='cartesian'
            coordinates='sphericalpolar'

            # choose central point to calculate spherical polars around - if commented out defaults to [0,0,0]
            spherical_bragg_vec=[1.35,1.42,0.96] 


            # choose to change frame of reference for mapping from default ‘hkl’ to 'qxqyqz' :
            frame_name='hkl'




