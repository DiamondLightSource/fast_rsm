Calcuations within fast_rsm mapper
===================================

full reciprocal space map
---------------------------

steps in code

- calculate volume start,stop,steps
- for each scan 
    - start pool of processes
    - chunk up indices
        - give each chunk to bin_map_with_indices
        - incides are either skipped or given to bin_one_map
        - gets q_vectors for the images at that index
        - then performs weighted 3d bin using mapper_c_utils.weighted_bin_3d
    - each chunk gives a shared_memory_volume,shared_memory_counts
    - read in these shared memorys, then sum along axis
    - get normalised map by dividing volume/shared_memory_counts


getting   q vectors for each pixel in image
.............................................

calculate the position of detector , along with vertical and horizontal direction of pixels on the detector
- diffraction_utils - applies delta then gamma to incident [0,0,1]
- use the detector vector to calculate rotations to apply to horizontal an vertical directions



calculate each exit vector using a combination of sample-detector distance vector, and vertical and horizontal pixel displacement vectors
normalise these exit vectors
calculate linear polarisation correction for exit vectors - using mapper_c_utils.linear_pol_correction
combines exit vector with incident vector to get the Q vectors

if requested in HKL use inverse UB matrix to convert from raw Q into HKL


weighted 3d bin using mapper_c_utils.weighted_bin_3d
........................................................

uses start,stop,step to determine if the pixel q vector is within the bounds defined




GIWAXS  
========

static 
-------
calculate qlimits  - gets high and low pix distance for each direction, then combine with detector distance to calculate maximum and minimum exit angle. Use k vector values and incident angle values to get upper and lower q values. 
alter beam centre if vertical setup used. 
create a poni if one is not provided - assuming orthogonal detector 
calculate binning resolution in Q if no qmap bins given

for q_parallel Vs q_perpendicular
    use  units.get_unit_fiber()  to set inplane and out of plane units - which accounts for incident angle, has sample_orientation=1
        unit_qip_name = "qip_A^-1"
        unit_qoop_name = "qoop_A^-1"
    get gamma and delta values, and then convert to cartesian rot1,rot2,rot3
    update PONI rotations
    use ai.integrate2d() with method=("no", "csr", "cython")

for exit angles
    use  units.get_unit_fiber()  to set inplane and out of plane units - which accounts for incident angle, has sample_orientation=1
        unit_qip_name ="exit_angle_horz_deg"
        unit_qoop_name = "exit_angle_vert_deg"
    get gamma and delta values, and then convert to cartesian rot1,rot2,rot3
    update PONI rotations
    use ai.integrate2d() with method=("no", "csr", "cython")   

for I vs Q
    get gamma and delta values, and then convert to cartesian rot1,rot2,rot3
    update PONI rotations
    use ai.integrate1d_ng()  twice, once with unit="2th_deg" and once with unit="q_A^-1"


moving
========


for q_parallel Vs q_perpendicular
    go through all scans and calculate q limits, giving overall upper and lower qlimits for the collection of scans

    if using dcd adjust gammadata
    chunk up indices (ignoring skipped values)
    send of chunks to pyfai_move_qmap_worker
        use  units.get_unit_fiber()  to set inplane and out of plane units - which accounts for incident angle, has sample_orientation=1
            unit_qip_name = "qip_A^-1"
            unit_qoop_name = "qoop_A^-1"
        get gamma and delta values, and then convert to cartesian rot1,rot2,rot3
        update PONI rotations
        use ai.integrate2d() with method=("no", "csr", "cython")
        individually done and then added to totals, then totals are given to shared memory


for I vs q
    calculate maximum and minimum exitanlges