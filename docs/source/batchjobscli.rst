Batch processing multiple jobs from the command line
=====================================================

processing multiple scans independently from command line might be required, specially if post-processing bulk experimental data.

This can be done using a bash script. 

#. open a terminal, follow the instruction above, prepare your setup file and make sure you are able to process at least one of the scans you intend to batch process
#. create a bash script in the same form of the one below (modify PATH_TO_EXP and SCANLIST)
#. execute the bash script from terminal using the command ./name_of_bash_script.sh

.. code-block:: bash

        #!/bin/bash

        PATH_TO_EXP='/path_to_setup_file/set_up_file.py'
        SCANLIST=(550483 550492 550493)
        VERSION='testing'

        cd /dls_sw/apps/fast_rsm/$VERSION/fast_rsm/CLI/i07/

        for scan in  "${SCANLIST[@]}"; do

            echo "Starting processing for scan: $scan"
            python runconfig.py -exp $PATH_TO_EXP -s $scan
            exit_code=$?
            echo "Python script for scan $scan exited with code $exit_code" || true
            if [ $exit_code -ne 0 ]; then
                echo "Error processing scan $scan, continuing with next scan"
            fi
            echo "Finished processing scan: $scan"

        done