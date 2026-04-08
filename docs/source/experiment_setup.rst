walkthroughs for creating the experiment setup file
=====================================================


To process your collected data you will need to create an experimental setup file, so that the fast_rsm software understands both the geometry of the setup as well as the output you would like it to calculate.

A template experiment setup file can be opened from a new terminal using the command:

    .. code:: python

        module load fast_rsm
        makesetup

This will open up a copy of the experiment setup file template, which you can then edit with your experimental information. 

Use the walkthrough guides below to create your exp_setup.py file. 

.. toctree::
   :maxdepth: 1

   minimum_setup
   optional_setup_all
   optional_setup_giwaxs

