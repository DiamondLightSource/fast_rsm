Setting up Linux session and wilson connection
==================================================

Firstly login to a diamond linux machine. If you're on the beamline, the beamline scientists will show you which machine uses linux. 

If you're at not at Diamond, use nomachine to access a linux workstation with access to the diamond servers. A guide on setting up NoMachine can be found here:

 * https://www.diamond.ac.uk/Users/Experiment-at-Diamond/IT-User-Guide/Not-at-DLS/Nomachine.html 

Once you are logged into a linux machine, you will need to setup your SSH connection to wilson. To do this, open a new terminal , import the fast_rsm module and run the command ssh-instructions. This will give you a step-by-step guide to allow your SSH connection to the HPC cluster at Diamond. 

.. code:: bash

    module load fast_rsm
    ssh-instructions