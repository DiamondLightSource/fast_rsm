Examples using test data
==============================

Create the setup file
------------------------

Try creating your own experiment setup file using the details described below, then compare to the example setup file by clicking the expandable text underneath. 

experiment details:

* The experiment was conducted in experimental hutch 1 using a vertical setup
* The data is stored at the following path '/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/si30563' 
* the beam centre was (1428,978)
* the detector distance was 0.4264 meters
* Thee detector positioning system was not used
* Set the output mapped image to be 1200 x 1200 pixels
* use a radial step value of 0.01
* use the mask file at the following path  '/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/masks/Ayo_Mask_1.edf'
* set the outputs to be an intensity Vs Q profile
* set the map_per_image value to True



.. collapse:: ANSWER - click to view full example setup file

    .. code-block:: python

        setup = 'vertical'
        experimental_hutch=1
        local_data_path =   '/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/si30563'
        local_output_path = <your_chosen_mapped_data_outpath>
        beam_centre = (1428,978)
        detector_distance =0.4264
        qmapbins=(1200,1200)
        radialstepval=0.01
        edfmaskfile= '/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/masks/Ayo_Mask_1.edf'
        process_outputs=['pyfai_ivsq']
        map_per_image = True


run the processing
---------------------

Use your newly created setup file from the first step, to run the processing job from the command line interface. 
Open a new terminal and do the following commands:

.. code-block:: python

    module load fast_rsm
    process_scans -exp <path_to_your_new_experiment_setup_file> -s 432196

If everything has worked correctly you should see an output in the terminal similar to the following:

.. code-block:: bash

    (/dls_sw/apps/fast_rsm/v2.0.0/conda_env) [fedid@computer ~]$ process_scans -exp /dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/setups/shortened_example_si30563.py -s 432196
    /home/fedid/mapscript_2025-11-12_14h03m26s.sh

    Submitted batch job 23943277
    Slurm output file: /home/fedid/fast_rsm//slurm-23943277.out 


    ***********************************
    ***STARTING TO MONITOR TAIL END OF FILE, TO EXIT THIS VIEW PRESS ANY LETTER FOLLOWED BY ENTER**** 
    *********************************** 

    /dls_sw/apps/fast_rsm/v2.0.0/conda_env/lib/python3.10/site-packages/diffraction_utils/io.py:90: MissingMetadataWarning: _parse_attenuation_filters_moving failed to parse a value, so its value will default to None.
    warn(MissingMetadataWarning(
    config values loaded
    Took 1.0269806385040283s to load all nexus files.
    starting process pool with num_threads=40
    finished process pool
    saved 2d Qmap data to            /dls/science/users/fedid/output//Qmap_432196_2025-11-12_14h03m36s.hdf5

    2d Qmap calculations took 0.03856806357701619 minutes
    PROCESSING FINISHED.
    Target phrase 'PROCESSING FINISHED' found. Closing tail.

