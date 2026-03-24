
Using fast_rsm at Diamond Light Source
=======================================

Below are detailed instructions on how to use fast_rsm at i07. If you requied any extra assistance, ask a beamline member of staff or email philip.mousley@diamond.ac.uk

.. toctree::
   :maxdepth: 2

   nomachine_wilson


Once you have a diamond linux session and your SSH connection setup, follow the steps below to run your processing jobs. 

.. toctree::
   :maxdepth: 2

   experiment_setup
   processcli
   batchjobscli

Alternatively there is the option to setup your data collection macro so that the analysis is performed automatically:

.. toctree::
   :maxdepth: 2

   jobsfromgda


There are also a set of useful functions available from the command line one the fast_rsm module has been loaded

.. toctree::
   :maxdepth: 2

   extra_cli


Debugging
--------------
    #. If you can't get nomachine working, contact the beamline staff/diamond IT support.
    #. If module load fast_rsm doesn't work:
        a. if you get "command not found: module", you're probably on your own computer, not the diamond servers!
        b. If you get "Unable to locate a modulefile for 'fast_rsm', there is a serious problem. Contact beamline staff immediately.
    #. The slurmout file should contain error messages with information about what error was encountered. If these do not provide an easy solution, forward this onto the i07's data analysis scientist for support

    Q: I finished processing, but the output looks ridiculous. What do I do?

    There are many reasons why this could happen, but first make sure that you remembered to correctly set your central pixel (this is the easiest thing to forget). Otherwise, either some other piece of information has been entered incorrectly, or your data file is corrupt in some way. In case of the latter, contact the i07's data analysis scientist to investigate.



