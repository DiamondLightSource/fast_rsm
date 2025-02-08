
"""
This section prepares the calculation. You probably shouldn't change any'qperp_qpara_map',thing here
unless you know what you're doing.
"""

# Warn if dps offsets are silly.
if ((dpsx_central_pixel > 10) or (dpsy_central_pixel > 10) or 
    (dpsz_central_pixel > 10)):
    raise ValueError("DPS central pixel units should be meters. Detected "
                     "values greater than 10m")


# Which synchrotron axis should become the out-of-plane (001) direction.
# Defaults to 'y'; can be 'x', 'y' or 'z'.
if setup == 'vertical':
    oop = 'x'
elif setup == 'horizontal':
    oop = 'y'
elif setup == 'DCD':
    oop = 'y'
else:
    raise ValueError(
        "Setup not recognised. Must be 'vertical', 'horizontal' or 'DCD.")

# # Overwrite the above oop value depending on requested cylinder axis for polar
# # coords.
# if cylinder_axis is not None:
#     oop = cylinder_axis

if output_file_size > 2000:
    raise ValueError("output_file_size must not exceed 2000. "
                     f"Value received was {output_file_size}.")

# Max number of cores available for processing.
num_threads = multiprocessing.cpu_count()

data_dir = Path(local_data_path)

processing_dir = Path(local_output_path)

# Here we calculate a sensible file name that hasn't been taken.
i = 0

save_file_name = f"mapped_scan_{scan_numbers[0]}_{i}"
save_path = processing_dir / save_file_name
# Make sure that this name hasn't been used in the past.
while (os.path.exists(str(save_path) + ".npy") or
       os.path.exists(str(save_path) + ".vtk") or
       os.path.exists(str(save_path) + "_l.txt") or
       os.path.exists(str(save_path) + "_tth.txt") or
       os.path.exists(str(save_path) + "_Q.txt") or
       os.path.exists(save_path)):
    i += 1
    save_file_name = f"mapped_scan_{scan_numbers[0]}_{i}"
    save_path = processing_dir / save_file_name

    if i > 1e7:
        raise ValueError(
            "Either you tried to save this file 10000000 times, or something "
            "went wrong. I'm going with the latter, but exiting out anyway.")

from datetime import datetime
# Work out the paths to each of the nexus files. Store as pathlib.Path objects.
nxs_paths = [data_dir / f"i07-{x}.nxs" for x in scan_numbers]


# # The frame/coordinate system you want the map to be carried out in.
# # Options for frame_name argument are:
# #     Frame.hkl     (map into hkl space - requires UB matrix in nexus file)
# #     Frame.sample_holder   (standard map into 1/Å)
# #     Frame.lab     (map into frame attached to lab.)
# #
# # Options for coordinates argument are:
# #     Frame.cartesian   (normal cartesian coords: hkl, Qx Qy Qz, etc.)
# #     Frame.polar       (cylindrical polar with cylinder axis set by the
# #                        cylinder_axis variable)
# #
# # Frame.polar will give an output like a more general version of PyFAI.
# # Frame.cartesian is for hkl maps and Qx/Qy/Qz. Any combination of frame_name
# # and coordinates will work, so try them out; get a feel for them.
# # Note that if you want something like a q_parallel, q_perpendicular projection,
# # you should choose Frame.lab with cartesian coordinates. From this data, your
# # projection can be easily computed.
# frame_name = Frame.hkl
# coordinates = Frame.cartesian

# # Ignore this unless you selected Frame.polar.
# # This sets the axis about which your polar coordinates will be generated.
# # Options are 'x', 'y' and 'z'. These are the synchrotron coordinates, rotated
# # according to your requested frame_name. For instance, if you select
# # Frame.lab, then 'x', 'y' and 'z' will correspond exactly to the synchrotron
# # coordinate system (z along beam, y up). If you select frame.sample_holder and
# # rotate your sample by an azimuthal angle µ, then 'y' will still be vertically
# # up, but 'x' and 'z' will have been rotated about 'y' by the angle µ.
# # Leave this as "None" if you aren't using cylindrical coordinates.
cylinder_axis = None


# # Construct the Frame object from the user's preferred frame/coords.
# map_frame = Frame(frame_name=frame_name, coordinates=coordinates)
#projected2d==None
# Prepare the pixel mask. First, deal with any specific pixels that we have.
# Note that these are defined (x, y) and we need (y, x) which are the
# (slow, fast) axes. So: first we need to deal with that!
if specific_pixels is not None:
    specific_pixels = specific_pixels[1], specific_pixels[0]

# Now deal with any regions that may have been defined.
# First make sure we have a list of regions.
# if isinstance(mask_regions, Region):
#     mask_regions_list = [mask_regions]
# else:
#     mask_regions_list=[Region(*maskval) for maskval in mask_regions]
mask_regions_list=[]
if mask_regions !=None:
    mask_regions_list=[maskval if isinstance(maskval,Region) else Region(*maskval) for maskval in mask_regions]

# Now swap (x, y) for each of the regions.
if mask_regions_list is not None:
    for region in mask_regions_list:
        region.x_start, region.y_start = region.y_start, region.x_start
        region.x_end, region.y_end = region.y_end, region.x_end

# Finally, instantiate the Experiment object.
experiment = Experiment.from_i07_nxs(
    nxs_paths,beam_centre, detector_distance, setup, 
    using_dps=using_dps,experimental_hutch=experimental_hutch)

experiment.mask_pixels(specific_pixels)
experiment.mask_edf(edfmaskfile)
experiment.mask_regions(mask_regions_list)
experiment.setup=setup
if 'savetiffs' in globals():
    experiment.savetiffs=savetiffs
else:
    experiment.savetiffs=False

if 'savedats' in globals():
    experiment.savedats=savedats
else:
    experiment.savedats=False

if 'qmapbins' not in globals():
    qmapbins=0


"""
This section is for changing metadata that is stored in, or inferred from, the
nexus file. This is generally for more nonstandard stuff.
"""

total_images = 0
for i, scan in enumerate(experiment.scans):
    total_images += scan.metadata.data_file.scan_length
    # Deal with the dps offsets.
    if scan.metadata.data_file.using_dps:
        if scan.metadata.data_file.setup == 'DCD':
            # If we're using the DCD and the DPS, our offset calculation is
            # somewhat involved. If you're confused about this and would like to
            # see a derivation, contact Richard Brearton.

            # Work out the in-plane and out-of-plane incident light angles.
            # To do this, first grab a unit vector pointing along the beam.
            lab_frame = Frame(Frame.lab, scan.metadata.diffractometer, 
                              coordinates=Frame.cartesian)
            beam_direction = scan.metadata.diffractometer.get_incident_beam(
                lab_frame).array

            # Now do some basic handling of spherical polar coordinates.
            out_of_plane_theta = np.sin(beam_direction[1])
            cos_theta_in_plane = beam_direction[2]/np.cos(out_of_plane_theta)
            in_plane_theta = np.arccos(cos_theta_in_plane)

            # Work out the total displacement from the undeflected beam of the
            # central pixel, in the x and y directions (we know z already).
            # Note that dx, dy are being calculated with signs consistent with
            # synchrotron coordinates.
            total_dx = -detector_distance * np.tan(in_plane_theta)
            total_dy = detector_distance * np.tan(out_of_plane_theta)

            # From these values we can compute true DPS offsets.
            dps_off_x = total_dx - dpsx_central_pixel
            dps_off_y = total_dy - dpsy_central_pixel

            scan.metadata.data_file.dpsx += dps_off_x
            scan.metadata.data_file.dpsy += dps_off_y
            scan.metadata.data_file.dpsz -= dpsz_central_pixel
        else:
            # If we aren't using the DCD, our life is much simpler.
            scan.metadata.data_file.dpsx -= dpsx_central_pixel
            scan.metadata.data_file.dpsy -= dpsy_central_pixel
            scan.metadata.data_file.dpsz -= dpsz_central_pixel

        # Load from .dat files if we've been asked.
        if load_from_dat:
            dat_path = data_dir / f"{scan_numbers[i]}.dat"
            scan.metadata.data_file.populate_data_from_dat(dat_path)

    # This is where you might want to overwrite some data that was recorded
    # badly in the nexus file. See (commented out) examples below.
    # scan.metadata.data_file.probe_energy = 12500
    # scan.metadata.data_file.transmission = 0.4
    # scan.metadata.data_file.using_dps = True
    # scan.metadata.data_file.ub_matrix = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])

    #reads in skip information and skips specified images in specified files
    
    if ('skipscans' in globals()):
        if (int(scan_numbers[i]) in skipscans):
            scan.skip_images+=skipimages[np.where(np.array(skipscans)==int(scan_numbers[i]))[0][0]]

    # """

if 'qmapbins' not in globals():
    qmapbins=0
import os,sys

# Get the full path of the current file
full_path = __file__

f =open(full_path)
joblines=f.readlines()
f.close()
pythonlocation=sys.executable

#grab ub information 
ubinfo=[scan.metadata.data_file.nx_instrument.diffcalchdr for scan in experiment.scans]



# """
# This section contains all of the logic for running the calculation. 
# If calculating a full map you shouldn't run this on your local computer,
#    it'll either raise an exception or take
# forever.
# """
from fast_rsm.diamond_utils import save_binoculars_hdf5
from time import time
import nexusformat.nexus as nx
import h5py
        

for i, scan in enumerate(experiment.scans):
    start_time = time()
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    name_end=scan_numbers[i]
    GIWAXS_names=['curved_projection_2D','pyfai_1D','qperp_qpara_map' ,'large_moving_det','pyfai_2dqmap_IvsQ']
    GIWAXScheck=np.isin(GIWAXS_names,process_outputs)
    if GIWAXScheck.sum()>0:
        projected2d=None
        projected_name=f'GIWAXS_{name_end}_{datetime_str}'
        hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        PYFAI_MASK=edfmaskfile
        if 'large_moving_det' in process_outputs:
            process_start_time=time()
            experiment.load_curve_values(scan)
            PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)

            experiment.pyfaidiffractometer(hf,scan, num_threads,  local_output_path,PYFAI_PONI,radialrange,radialstepval,qmapbins)

   
            print(f"saved 2d map and 1D integration data to {local_output_path}/{projected_name}.hdf5")           
           
            total_time = time() - process_start_time
            print(f"\n large_moving_det calculation took {total_time}s")


        if 'pyfai_2dqmap_IvsQ' in process_outputs:
            process_start_time=time()
            experiment.load_curve_values(scan)
            PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
            experiment.pyfai_static_diff(hf,scan, num_threads,  local_output_path,PYFAI_PONI,ivqbins,qmapbins)
            print(f"saved 2d map and 1D integration data to {local_output_path}/{projected_name}.hdf5")
            total_time = time() - process_start_time
            print(f"\n Azimuthal integration 2d took {total_time}s")
            
        if 'curved_projection_2D' in process_outputs:
            process_start_time=time()
            projected2d=experiment.curved_to_2d(scan)
            PYFAI_PONI=experiment.createponi(local_output_path,experiment.projshape,offset=experiment.vertoffset)
            twothetas,Qangs,intensities,config= experiment.pyfai1D(local_data_path,PYFAI_MASK,PYFAI_PONI,\
                                local_output_path,scan,projected2d=projected2d)
            experiment.save_projection(hf,projected2d,twothetas,Qangs,intensities,config)
    
            print(f"saved projection to {local_output_path}/{projected_name}.hdf5")
            
            total_time = time() - process_start_time
            print(f"\nProjecting 2d took {total_time}s")


           
        if 'pyfai_1D' in process_outputs:
           
       
            experiment.load_curve_values(scan)
            name_end=scan_numbers[i]
            #image2dshape=experiment.scans[i].metadata.data_file.image_shape
            PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
            twothetas,Qangs,intensities,config= experiment.pyfai1D(local_data_path,PYFAI_MASK,PYFAI_PONI,\
                              local_output_path,scan)
            print(np.max(intensities))
            experiment.save_integration(hf,twothetas,Qangs,intensities,config,scan)
   
           
            print(f'saved 1D profile to {local_output_path}/{projected_name}.hdf5')
   

           
        if 'qperp_qpara_map' in process_outputs:
            frame_name = Frame.lab
            coordinates = Frame.cartesian
            map_frame = Frame(frame_name=frame_name, coordinates=coordinates)
            name_end=scan_numbers[i]
            qperp_qpara_map=experiment.calc_qpara_qper(scan,oop, map_frame,proj2d=projected2d)
            experiment.save_qperp_qpara(hf, qperp_qpara_map,scan)
            print(f'saved qperp_qpara_map to {local_output_path}/{projected_name}.hdf5')
   
   
            #print('Finished qperp_qpara mapping')
        experiment.save_config_variables(hf,joblines,pythonlocation)
        hf.close()
        print(f'finished processing scan {name_end}')

#for i, scan in enumerate(experiment.scans):
if ('pyfai_qmap' in process_outputs)&(map_per_image==True):
    for i, scan in enumerate(experiment.scans):
        name_end=scan_numbers[i]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name=f'Qmap_{name_end}_{datetime_str}'
        hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        process_start_time=time()
        experiment.load_curve_values(scan)
        PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
        experiment.pyfai_static_qmap(hf,scan, num_threads,local_output_path,PYFAI_PONI,ivqbins,qmapbins)
        experiment.save_config_variables(hf,joblines,pythonlocation,globals())
        hf.close()
        print(f"saved 2d map  data to {local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time
        print(f"\n 2d Q map calculations took {total_time}s")
    
if ('pyfai_qmap' in process_outputs)&(map_per_image==False):
    #for i, scan in enumerate(experiment.scans):
    scanlist=experiment.scans
    name_end=scan_numbers[0]
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    projected_name=f'Qmap_{name_end}_{datetime_str}'
    hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
    process_start_time=time()
    experiment.load_curve_values(scanlist[0])
    PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
    experiment.pyfai_moving_qmap(hf,scanlist, num_threads,  local_output_path,PYFAI_PONI,radialrange,radialstepval,qmapbins)
    experiment.save_config_variables(hf,joblines,pythonlocation,globals())
    hf.close()
    print(f"saved 2d map data to {local_output_path}/{projected_name}.hdf5")           
    
    total_time = time() - process_start_time
    print(f"\n 2d Q map calculation took {total_time}s")


if ('pyfai_ivsq' in process_outputs)&(map_per_image==True):
    for i, scan in enumerate(experiment.scans):
        name_end=scan_numbers[i]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name=f'IvsQ_{name_end}_{datetime_str}'
        hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        process_start_time=time()
        experiment.load_curve_values(scan)
        PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
        experiment.pyfai_static_ivsq(hf,scan, num_threads,local_output_path,PYFAI_PONI,ivqbins,qmapbins)
        experiment.save_config_variables(hf,joblines,pythonlocation,globals())
        hf.close()
        print(f"saved 1d integration data to {local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time 
        print(f"\n Azimuthal integrations took {total_time}s")
    
if ('pyfai_ivsq' in process_outputs)&(map_per_image==False):
    for i, scan in enumerate(experiment.scans):
        name_end=scan_numbers[i]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name=f'IvsQ_{name_end}_{datetime_str}'
        hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        process_start_time=time()
        experiment.load_curve_values(scan)
        PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
        experiment.pyfai_moving_ivsq(hf,scan, num_threads,local_output_path,PYFAI_PONI,radialrange,radialstepval,qmapbins)
        experiment.save_config_variables(hf,joblines,pythonlocation,globals())
        hf.close()
        print(f"saved 1d integration data to {local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time 
        print(f"\n Azimuthal integration took {total_time}s")

if ('pyfai_exitangles' in process_outputs)&(map_per_image==True):
    for i, scan in enumerate(experiment.scans):
        name_end=scan_numbers[i]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name=f'exitmap_{name_end}_{datetime_str}'
        hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        process_start_time=time()
        experiment.load_curve_values(scan)
        PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
        experiment.pyfai_static_exitangles(hf,scan, num_threads,PYFAI_PONI,ivqbins,qmapbins)
        experiment.save_config_variables(hf,joblines,pythonlocation,globals())
        hf.close()
        print(f"saved 2d exit angle map  data to {local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time
        print(f"\n 2d exit angle map calculations took {total_time}s")
        
if ('pyfai_exitangles' in process_outputs)&(map_per_image==False):
    for i, scan in enumerate(experiment.scans):
        name_end=scan_numbers[i]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name=f'exitmap_{name_end}_{datetime_str}'
        hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        process_start_time=time()
        experiment.load_curve_values(scan)
        PYFAI_PONI=experiment.createponi(local_output_path,experiment.imshape,beam_centre=experiment.beam_centre)
        experiment.pyfai_moving_exitangles(hf,scan, num_threads, local_output_path, PYFAI_PONI,radialrange,radialstepval,qmapbins)
        experiment.save_config_variables(hf,joblines,pythonlocation,globals())
        hf.close()
        print(f"saved 2d exit angle map  data to {local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time
        print(f"\n 2d exit angle map calculations took {total_time}s")
        
        

    print(f'finished processing scan {name_end}')
        
    

    
if 'full_reciprocal_map' in process_outputs:
    frame_name = Frame.hkl
    coordinates = Frame.cartesian
    map_frame = Frame(frame_name=frame_name, coordinates=coordinates)
    start_time = time()
    # Calculate and save a binned reciprocal space map, if requested.
    experiment.binned_reciprocal_space_map(
        num_threads, map_frame, output_file_size=output_file_size, oop=oop,
        min_intensity_mask=min_intensity,
        output_file_name=save_path, 
        volume_start=volume_start, volume_stop=volume_stop,
        volume_step=volume_step,
        map_each_image=map_per_image)

    if save_binoculars_h5==True:
        outvars=globals()
        
        save_binoculars_hdf5(str(save_path) + ".npy", str(save_path) + '.hdf5',joblines,pythonlocation,outvars)
        print(f"\nSaved BINoculars file to {save_path}.hdf5.\n")

    # Finally, print that it's finished We'll use this to work out when the
    # processing is done.
    total_time = time() - start_time
    print(f"\nProcessing took {total_time}s")
    print(f"This corresponds to {total_time*1000/total_images}ms per image.\n")
    
print("PROCESSING FINISHED.")
