Using fast_rsm at Diamond Light Source
=======================================

Below are detailed instructions on how to use fast_rsm at i07. If you requied any extra assistance, ask a beamline member of staff or email philip.mousley@diamond.ac.uk


Firstly follow the steps given by beamline staff or data analysis contact to setup SSH connection to wilson. Once your SSH connection has been setup, follow the steps below to run your processing jobs. 

.. toctree::
   :maxdepth: 2

   diamondconnect
   experimentfiles
   processcli
   batchjobscli

Alternatively there is the option to setup your data collection macro so that the analysis is performed automatically:

.. toctree::
   :maxdepth: 2

   jobsfromgda


Useful CLI functions
--------------------

Once the fast_rsm module has been loaded there are also some other useful functions made accessible
    **makesetup** - use this to open up a template experiment_setup file, edit with your experiment information and then save to your preferred directory


    **makemask**  - use this to quickly make a mask for you processing. It will open up the pyFAI mask creation tool and load in the first image of your scan. From here you can use the GUI tools to draw on the desired mask and then save to a specified directory. 

    **pyFAI-calib2** - use this to launch the :py:mod:`pyFAI` calibration tool which you can use to analyse your calibration data to create a PONI file - see the :py:mod:`pyFAI` documentation page on the `calibration GUI`_ for more details



    **get_setups** -  use this to get setup information used to calculate a hdf5 file. if hdf5 has the configuration saved to config, and it contains the joblines from the analysis job sent to the cluster - then you can type 

    .. code-block:: bash

        get_setups -hf  path/to/hdf5file.hdf5   -outdir   path/to/out/directory

    This will open up the hdf5 file, locate the joblines information, and then it will save a copy if this information and open exp_setup.py and calc_setup.py files with the information that was used to send off the analysis job

Debugging
--------------
    #. If you can't get nomachine working, contact the beamline staff/diamond IT support.
    #. If module load fast_rsm doesn't work:
        a. if you get "command not found: module", you're probably on your own computer, not the diamond servers!
        b. If you get "Unable to locate a modulefile for 'fast_rsm', there is a serious problem. Contact beamline staff immediately.
    #. The slurmout file should contain error messages with information about what error was encountered. If these do not provide an easy solution, forward this onto the i07's data analysis scientist for support

    Q: I finished processing, but the output looks ridiculous. What do I do?

    There are many reasons why this could happen, but first make sure that you remembered to correctly set your central pixel (this is the easiest thing to forget). Otherwise, either some other piece of information has been entered incorrectly, or your data file is corrupt in some way. In case of the latter, contact the i07's data analysis scientist to investigate. It is usually possible, with some work, to recover your data.



.. _calibration GUI: https://pyfai.readthedocs.io/en/stable/usage/cookbook/calib-gui/index.html#cookbook-calibration-gui


    