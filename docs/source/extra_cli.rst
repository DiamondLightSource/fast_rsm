

Useful CLI functions
=======================

Once the fast_rsm module has been loaded there are also some other useful functions made accessible
    
makesetup
------------

Use this to open up a template experiment_setup file, edit with your experiment information and then save to your preferred directory


makemask
---------------

use this to quickly make a mask for you processing. It will open up the pyFAI mask creation tool and load in the first image of your scan. From here you can use the GUI tools to draw on the desired mask and then save to a specified directory. 

pyFAI-calib2
-----------------

use this to launch the :py:mod:`pyFAI` calibration tool which you can use to analyse your calibration data to create a PONI file - see the :py:mod:`pyFAI` documentation page on the `calibration GUI`_ for more details


get_setups
---------------

use this to get setup information used to calculate a hdf5 file. If the hdf5 file has the configuration saved to config, and it contains the joblines from the analysis job sent to the cluster - then you can type 

.. code-block:: bash

    get_setups -hf  path/to/hdf5file.hdf5   -outdir   path/to/out/directory

This will open up the hdf5 file, locate the joblines information, and then it will save a copy if this information and open exp_setup.py and calc_setup.py files with the information that was used to send off the analysis job


tiffconvert
-------------

.. note::

    Currently only available on the testing branch of fast-rsm

Open a new terminal on any of the diamond linux machines
Load the testing version of the fast_rsm module using :     

.. code-block:: bash

    module load fast_rsm/testing

The extractor can then be used from the command line using one of the following structures:   
To specify a few scan numbers at a time, use the -s option followed by the scan numbers e.g. :
    
    .. code-blck:: bash
    
    tiffextract -dir  /path/to/data/directory -s 1234  1235 
    
To specify a scan range of numbers evenly spaces use the -sr followed by double brackets with start,stop,step e.g.

.. code-block:: bash

    tiffextract -dir /path/to/data/directory -sr [[1120, 1150, 5]]  
    
To specify extraction of all .nxs file in the directory use -a flag e.g.

.. code-block:: bash

    tiffextract -dir /path/to/data/directory -a

The default output folder is the  experiment     /processing  directory, however you can add an optional different output folder by adding -o followed by the new output path e.g. 

.. code-block:: bash

    tiffextract -dir  /path/to/data/directory -s 1234  1235  -o /path/to/new/output

.. _calibration GUI: https://pyfai.readthedocs.io/en/stable/usage/cookbook/calib-gui/index.html#cookbook-calibration-gui
