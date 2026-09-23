
Calculations for GIWAXS datasets
=================================

Scans in GIWAXS experiments fall into two categories:
 - Static scans - collect data with the detector stationary 
 - Moving scans - collect data with detector moving

The analysis output is one of three options
 - Intensity Vs Q profile (1D)
 - Q parallel Vs Q q_perpendicular (2D)
 - Exit angle in-plane Vs exit angle out-of-plane (2D)

below are the analysis steps used for each type of scan, depending on the analysis output required


static scan -  I vs Q
-------------------------
    1. initiate a standard AzimuthalIntegrator object
        .. code-block:: python

            aistart = pyFAI.load(pyfaiponi)

    2. get gamma and delta values, convert to cartesian rot1,rot2,rot3 and then update rot values in PONI
    3. use `ai.integrate1d_ng`_  with the standard two-theta unit and then convert to Q-values to get three column tth,Q,I dataset:

        .. code-block:: python

            tth, I = my_ai.integrate1d_ng(img_data,
                                  ivqbins,
                                  unit="2th_deg", polarization_factor=1)
            Q= [experiment.calcq(tthval, experiment.incident_wavelength) for tthval in tth]
   

static scan - :math:`Q_{para}` Vs :math:`Q_{perp}`
--------------------------------------------------------
    1. calculate qlimits using `experiment.calcqlim`_ - gets the maximum and mimium pixel distances for each vertical and horizontal direction, then combines with detector distance to calculate maximum and minimum exit angle. From exit angles calculates full range of q values. 
    2. alter beam centre if vertical setup used. 
    3. create a poni file if one is not provided - assuming orthogonal detector 
    4. calculate binning resolution in Q if no qmap bins given

    5. use  `units.get_unit_fiber`_  to set inplane and out of plane units, which account for incident angle, and have sample_orientation=1:
        .. code-block:: python
            
            unit_qip_name = "qip_A^-1"
            unit_qoop_name = "qoop_A^-1"
    
    6. initiate a FiberIntegrator object using poni file
        .. code-block:: python

            aistart = pyFAI.load(pyfaiponi, type_="pyFAI.integrator.fiber.FiberIntegrator")
    7. get gamma and delta values, convert to cartesian rot1,rot2,rot3 and then update rot values in PONI
    
    8. use  `ai.integrate2d`_ with method=("no", "csr", "cython")
        .. code-block:: python

            map2d = my_ai.integrate2d(img_data, qmapbins[0], qmapbins[1], unit=(unit_qip, unit_qoop),\
                                radial_range=(ai_limits[0],ai_limits[1]),azimuth_range=(ai_limits[2],ai_limits[3]), method=("no", "csr", "cython"))
 

static scan - Exit angles
---------------------------
    1. calculate angle limits using `experiment.calcanglim`_ - gets the maximum and mimium pixel distances for each vertical and horizontal direction, then combines with detector distance to calculate maximum and minimum exit angles.
    2. alter beam centre if vertical setup used. 
    3. create a poni file if one is not provided - assuming orthogonal detector 
   
    4. use  `units.get_unit_fiber`_  to set inplane and out of plane units, which account for incident angle, and have sample_orientation=1:
        .. code-block:: python
            
            unit_qip_name ="exit_angle_horz_deg"
            unit_qoop_name = "exit_angle_vert_deg"

    5. initiate a FiberIntegrator object
        .. code-block:: python

            aistart = pyFAI.load(pyfaiponi, type_="pyFAI.integrator.fiber.FiberIntegrator")

    6. get gamma and delta values, convert to cartesian rot1,rot2,rot3 and then update rot values in PONI
    
    7. use  `ai.integrate2d`_ with method=("no", "csr", "cython")
        .. code-block:: python

            map2d = my_ai.integrate2d(img_data, qmapbins[0],qmapbins[1], unit=(unit_qip, unit_qoop),\
                                radial_range=(ai_limits[0],ai_limits[1]),azimuth_range=(ai_limits[2],ai_limits[3]), method=("no", "csr", "cython"))

moving  -  I vs Q
-------------------
    1. calculate angle limits using `experiment.calcanglim`_ - gets the maximum and mimium pixel distances for each vertical and horizontal direction, then combines with detector distance to calculate maximum and minimum exit angles.
    2. alter beam centre if vertical setup used. 
    3. create a poni file if one is not provided - assuming orthogonal detector 
    4. chunk up scans into groups, create group of azimuthal integrators editing the rotations for each image
    5. create MultiGeometry object including all integrators, and calculate `Multigeometry_integrate1d`_ 

        .. code-block:: python

            mg = MultiGeometry( ais,  unit=unit_tth_ip, wavelength=experiment.incident_wavelength, radial_range=(radrange[0],radrange[1]))
            result1d = mg.integrate1d(img_data_list, ivqbins)

    6. after each chunk is done, update the total sum and total count arrays
    

moving - :math:`Q_{para}` Vs :math:`Q_{perp}`
----------------------------------------------------

    1. calculate angle limits using `experiment.calcqlim`_ - gets the maximum and mimium pixel distances for each vertical and horizontal direction, then combines with detector distance to calculate maximum and minimum exit angles.From exit angles calculates full range of q values.
    2. alter beam centre if vertical setup used. 
    3. create a poni file if one is not provided - assuming orthogonal detector 
    4. chunk up scans into groups, create group of azimuthal integrators editing the rotations for each image.
    5. use  `units.get_unit_fiber`_  to set inplane and out of plane units, which account for incident angle, and have sample_orientation=1:

        .. code-block:: python
            
            unit_qip_name = "qip_A^-1"
            unit_qoop_name = "qoop_A^-1"
    6. For each image in group do `ai.integrate2d`_ with method=("no", "csr", "cython")

        .. code-block:: python

            map2d = current_ai.integrate2d(current_img, qmapbins[0], qmapbins[1], unit=(unit_qip, unit_qoop),\
                                           radial_range=(ai_limits[0],ai_limits[1]),azimuth_range=(ai_limits[2],ai_limits[3]), method=("no", "csr", "cython"))
            
    7. after each chunk is done, update the total sum and total count arrays


moving -Exit angles
------------------------

    1. calculate angle limits using `experiment.calcanglim`_ - gets the maximum and mimium pixel distances for each vertical and horizontal direction, then combines with detector distance to calculate maximum and minimum exit angles.
    2. alter beam centre if vertical setup used. 
    3. create a poni file if one is not provided - assuming orthogonal detector 
    4. chunk up scans into groups, create group of azimuthal integrators editing the rotations for each image.
    5. use  `units.get_unit_fiber`_  to set inplane and out of plane units, which account for incident angle, and have sample_orientation=1:

        .. code-block:: python
            
            unit_qip_name ="exit_angle_horz_deg"
            unit_qoop_name = "exit_angle_vert_deg"
    6. For each image in group do `ai.integrate2d`_ with method=("no", "csr", "cython")

        .. code-block:: python

            map2d = current_ai.integrate2d(current_img, qmapbins[0], qmapbins[1], unit=(unit_qip, unit_qoop),\
                                           radial_range=(ai_limits[0],ai_limits[1]),azimuth_range=(ai_limits[2],ai_limits[3]), method=("no", "csr", "cython"))
            
    7. after each chunk is done, update the total sum and total count arrays




.. _units.get_unit_fiber: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.units.get_unit_fiber
.. _ai.integrate2d: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.integrator.azimuthal.AzimuthalIntegrator.integrate2d
.. _ai.integrate1d_ng: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.integrator.azimuthal.AzimuthalIntegrator.integrate1d_ng
.. _Multigeometry_integrate1d: https://pyfai.readthedocs.io/en/stable/api/pyFAI.html#pyFAI.multi_geometry.MultiGeometry.integrate1d
.. _experiment.calcanglim: ./apidocs/experiment/experiment.html#experiment.Experiment.calcanglim
.. _experiment.calcqlim: ./apidocs/experiment/experiment.html#experiment.Experiment.calcqlim