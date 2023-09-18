"""
This section prepares the calculation. You probably shouldn't change anything here
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

# Overwrite the above oop value depending on requested cylinder axis for polar
# coords.
if cylinder_axis is not None:
    oop = cylinder_axis

if output_file_size > 2000:
    raise ValueError("output_file_size must not exceed 2000. "
                     f"Value received was {output_file_size}.")

# Max number of cores available for processing.
num_threads = multiprocessing.cpu_count()

# Work out where the data is.
if local_data_path is None:
    data_dir = Path(f"/dls/i07/data/{year}/{experiment_number}/")
else:
    data_dir = Path(local_data_path)
# data_dir = Path(f"/Users/richard/Data/i07/{experiment_number}/")

# Store this for later.
if local_output_path is None:
    processing_dir = data_dir / "processing"
else:
    processing_dir = Path(local_output_path)
if data_sub_directory is not None:
    data_dir /= Path(data_sub_directory)

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


# Work out the paths to each of the nexus files. Store as pathlib.Path objects.
nxs_paths = [data_dir / f"i07-{x}.nxs" for x in scan_numbers]

# Construct the Frame object from the user's preferred frame/coords.
map_frame = Frame(frame_name=frame_name, coordinates=coordinates)

# Prepare the pixel mask. First, deal with any specific pixels that we have.
# Note that these are defined (x, y) and we need (y, x) which are the
# (slow, fast) axes. So: first we need to deal with that!
if specific_pixels is not None:
    specific_pixels = specific_pixels[1], specific_pixels[0]

# Now deal with any regions that may have been defined.
# First make sure we have a list of regions.
if isinstance(mask_regions, Region):
    mask_regions = [mask_regions]

# Now swap (x, y) for each of the regions.
if mask_regions is not None:
    for region in mask_regions:
        region.x_start, region.y_start = region.y_start, region.x_start
        region.x_end, region.y_end = region.y_end, region.x_end

# Finally, instantiate the Experiment object.
experiment = Experiment.from_i07_nxs(
    nxs_paths, beam_centre, detector_distance, setup, 
    using_dps=using_dps)

experiment.mask_pixels(specific_pixels)
experiment.mask_regions(mask_regions)

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

    # Would you like to skip any images in any scans? Do so here!
    # This shows how to skip the 9th in the 3rd scan (note the zero counting).
    # if i == 2:
    #     scan.skip_images.append(8)

"""
This section contains all of the logic for running the calculation. You shouldn't
run this on your local computer, it'll either raise an exception or take
forever.
"""
from fast_rsm.diamond_utils import save_binoculars_hdf5
from time import time


if __name__ == "__main__":
    start_time = time()
    # Calculate and save a binned reciprocal space map, if requested.
    experiment.binned_reciprocal_space_map(
        num_threads, map_frame, output_file_size=output_file_size, oop=oop,
        min_intensity_mask=min_intensity,
        output_file_name=save_path, 
        volume_start=volume_start, volume_stop=volume_stop,
        volume_step=volume_step,
        map_each_image=map_per_image)

    if save_binoculars_h5:
        save_binoculars_hdf5(str(save_path) + ".npy", str(save_path) + '.hdf5')
        print(f"\nSaved BINoculars file to {save_path}.hdf5.\n")

    # Finally, print that it's finished We'll use this to work out when the
    # processing is done.
    total_time = time() - start_time
    print(f"\nProcessing took {total_time}s")
    print(f"This corresponds to {total_time*1000/total_images}ms per image.\n")

    print("PROCESSING FINISHED.")
