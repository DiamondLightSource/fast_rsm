Connecting to Diamond system and loading fast_rsm
==================================================

#. Login to a diamond linux machine. If you're on the beamline, the beamline scientists will show you which machine uses linux. If you're at home, use nomachine [ https://www.diamond.ac.uk/Users/Experiment-at-Diamond/IT-User-Guide/Not-at-DLS/Nomachine.html ] to access a linux workstation with access to the diamond servers.
#. in the terminal enter 

    .. code-block:: bash

        module load fast_rsm

#. If this is your first time using the fast_rsm mapper, to setup a folder in your home directory enter

    .. code-block:: bash
        
        firstrun 

#.  Activate the fast_rsm conda environment by entering the command

    .. code-block:: bash

        activate