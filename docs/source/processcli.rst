Run a single process job from the command line 
===============================================

#. run the processing from the command line by typing  
    .. code-block:: bash
        
        process_scans -exp  'path/to/exp_setup.py' -s scan-numbers-to-be-mapped

    e.g.   
    
    .. code-block:: bash
        
        process_scans  -exp   /home/rpy65944/fast_rsm/example_exp_setup.py -s 441187 441188

#. alternatively use the -sr option to define an evenly spaced range of scans using the format [[start,stop,stepsize]]  **note  double brackets are needed even when specifying only one range**
    
    .. code-block:: bash

        -sr [[41187, 441189,1]]
#. use a list of lists to define several sets of evenly spaced scans using the format [[start1,stop1,stepsize1],[start2,stop2,stepsize2]],  where the ranges are inclusive i.e. the stop value is the final scan in the range which you want analysed
    .. code-block:: bash

        -sr [[41187, 441189,1] ,[41192, 441195,1]]