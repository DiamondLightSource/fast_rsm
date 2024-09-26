"""
This module contains parsers for different instruments that return Scan objects.
"""

from pathlib import Path
from typing import Union, Tuple, TYPE_CHECKING


from .diffractometers import I07Diffractometer

from . import scan
from .rsm_metadata import RSMMetadata

if TYPE_CHECKING:
    from .scan import Scan


import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Union
from warnings import warn


import nexusformat.nexus.tree as nx
import numpy as np
import pandas as pd
from nexusformat.nexus import nxload

from .data_file import DataFileBase
from .frame_of_reference import Frame
from .polarisation import Polarisation
from .region import Region
from .vector import Vector3



def from_i07(path_to_nx: Union[str, Path],
             beam_centre: Tuple[int],
             detector_distance: float,
             setup: str,
             path_to_data: str = '',
             using_dps: bool = False,
             experimental_hutch=0) -> 'Scan':
    """
    Instantiates a Scan from the path to an I07 nexus file, a beam centre
    coordinate tuple, a detector distance and a sample out-of-plane vector.

    Args:
        path_to_nx:
            Path to the nexus file containing the scan metadata.
        beam_centre:
            A (y, x) tuple of the beam centre, measured in the usual image
            coordinate system, in units of pixels.
        detector_distance:
            The distance between the sample and the detector.
        setup:
            What was the experimental setup? Can be "vertical", "horizontal"
            or "DCD".
        path_to_data:
            Path to the directory in which the images are stored. Defaults
            to '', in which case a bunch of reasonable directories will be
            searched for the images. This is useful in case you store the small
            .nxs file in a different place to the potentially very large image
            data (e.g. .nxs files on a local disc, .h5 files on portable hard
            drive).

    Returns:
        Corresponding instance of fast_rsm.scan.Scan
    """
    # Load the nexus file.
    i07_nexus = I07Nexus(path_to_nx, path_to_data,
                         detector_distance, setup, using_dps=using_dps,experimental_hutch=experimental_hutch)

    # Not used at the moment, but not deleted in case full UB matrix
    # calculations become important in the future (in which case we'll also
    # need to supply a second value).
    sample_oop = Vector3([0, 1, 0], Frame(Frame.sample_holder, None, None))

    # Load the state of the diffractometer; prepare the RSM metadata.
    diff = I07Diffractometer(i07_nexus, sample_oop, setup)
    metadata = RSMMetadata(diff, beam_centre)

    # Make sure that the sample_oop vector's frame's diffractometer is good.
    sample_oop.frame.diffractometer = diff

    return scan.Scan(metadata)



BAD_NEXUS_FILE = (
    "Nexus files suck. It turns out your nexus file sucked too. "
    "If you're seeing this message, it means some non-essential data couldn't "
    "be parsed by diffraction_utils.")


class BadNexusFileError(Exception):
    """
    Warns that a nexus file cannot be parsed because it is incorrectly
    formatted.
    """


class MissingMetadataWarning(UserWarning):
    """
    Warns a user that some metadata is missing.
    """


def _get_utf_8(string_like):
    """
    Takes something that might be a string, or might be something that we can
    decode into a utf-8 string. Tries to decode it as a utf-8 string.

    This function will raise a ValueError if it is passed something that isn't
    a string, at that cannot be decoded as a string.
    """
    # pylint: disable=raise-missing-from
    try:
        string_like = string_like.decode('utf-8')
    except AttributeError:
        if not isinstance(string_like, str):
            raise ValueError(
                "string_like object must be a string, or be able to be "
                f"decoded into a string. Instead got {string_like}.")
    return string_like


def warn_missing_metadata(func):
    """
    Wrap _parse functions with this to make them non-essential. If a parser's
    corresponding information is missing, a warning will be printed instead of
    an error being raised. This means that, if some DAQ dude fucked up and
    there's a load of data missing from the nexus file, at least someone can
    retroactively add in the data.Test addition message.
    """
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (nx.NeXusError, KeyError) as _:
            warn(MissingMetadataWarning(
                f"{func.__name__} failed to parse a value, so its value will "
                "default to None."))
    return inner_function


def i07_data_from_dat(path_to_dat: Union[str, Path]) -> pd.DataFrame:
    """
    Takes a path to an i07 .dat file and uses it to grab essential data. This
    function does not grab metadata from .dat files - that's what nexus files
    are for. However, sometimes nexus files break and we have to fall back on
    .dat files for reading data. This file returns a pandas dataframe containing
    all of the data and none of the metadata.

    Args:
        path_to_dat:
            The path to the .dat file containing the data.

    Returns:
        pandas.core.frame.DataFrame object containing all of the data in the
        table at the bottom of the .dat file.
    """
    # First find the line on which the metadata ends.
    with open(path_to_dat, 'r', encoding='utf-8') as open_dat:
        for line_number, line in enumerate(open_dat):
            if line.strip().endswith('&END'):
                skip_rows = line_number + 1

    return pd.read_table(path_to_dat, skiprows=skip_rows)


class NexusBase(DataFileBase):
    """
    This class contains *mostly* beamline agnostic nexus parsing convenience
    stuff. It's worth noting that this class still makes a series of assumptions
    about how data is laid out in a nexus file that can be broken. Instead of
    striving for some impossible perfection, this class is practical in its
    assumptions of how data is laid out in a .nxs file, and will raise if an
    assumption is violated. All instrument-specific assumptions that one must
    inevitably make to extract truly meaningful information from a nexus file
    are made in children of this class.

    Attrs:
        file_path:
            The local path to the file on the local filesystem.
        nxfile:
            The object produced by loading the file at file_path with nxload.
    """

    def __init__(self,
                 local_path: Union[str, Path],  # The path to this file.
                 local_data_path: Union[str, Path] = '',  # Path to the data.
                 locate_local_data=True):

        # Set up the nexus specific attributes.
        # This needs to be done *before* calling super().__init__!
        self.nxfile = nxload(local_path)
        self.nx_entry = self._parse_nx_entry()
        self.default_nx_data_name = self._parse_default_nx_data_name()
        self.default_nx_data = self._parse_default_nx_data()
        self.nx_instrument = self._parse_nx_instrument()
        self.nx_detector = self._parse_nx_detector()
        self.diamond_scan = self._parse_diamond_scan()
        self.scan_fields = self._parse_scan_fields()

        # Now we can call super().__init__ to run the remaining parsers.
        super().__init__(local_path, local_data_path, locate_local_data)

        # Finally, parse the motors.
        self.motors = self._parse_motors()

    def _parse_diamond_scan(self):
        """
        Parses the diamond_scan NXcollection inside the main NXentry.
        """
        return self.nx_entry["diamond_scan"]

    def _parse_scan_fields(self):
        """
        Returns the list of scan fields contained within the diamond scan.
        """
        # As of 09/2022, these are no longer guaranteed to be in a .nxs file,
        # so lets explicitly check and return an empty list if it's missing.
        if "scan_fields" not in self.diamond_scan:
            return []

        # Some explicit string casts in case these are byte arrays.
        try:
            return [x.decode('utf-8')
                    for x in self.diamond_scan["scan_fields"].nxdata]
        except AttributeError:
            # These are strings, not byte arrays, so we can just return now.
            return self.diamond_scan["scan_fields"].nxdata

    def _parse_nx_detector(self):
        """
        Returns the NXdetector instance stored in this NexusFile. This will
        need to be overridden for beamlines that put more than 1 NXdetector in
        a nexus file.

        Raises:
            ValueError if more than one NXdetector is found.
        """
        det, = self.nx_instrument.NXdetector
        return det

    def _parse_nx_instrument(self):
        """
        Returns the NXinstrument instanced stored in this NexusFile.

        Raises:
            ValueError if more than one NXinstrument is found.
        """
        instrument, = self.nx_entry.NXinstrument
        return instrument

    def _parse_nx_entry(self) -> nx.NXentry:
        """
        Returns this nexusfile's entry.

        Raises:
            ValueError if more than one entry is found.
        """
        entry, = self.nxfile.NXentry
        return entry

    def _parse_default_signal(self) -> np.ndarray:
        """
        The numpy array of intensities pointed to by the signal attribute in the
        nexus file.
        """
        # pylint: disable=bare-except
        try:
            return self.default_nx_data[self.default_signal_name].nxdata
        except:
            return BAD_NEXUS_FILE

    def _parse_default_axis(self) -> np.ndarray:
        """
        Returns the nxdata associated with the default axis.
        """
        # pylint: disable=bare-except
        try:
            return self.default_nx_data[self.default_axis_name].nxdata
        except:
            return BAD_NEXUS_FILE

    def _parse_default_signal_name(self):
        """
        Returns the name of the default signal.
        """
        # pylint: disable=bare-except
        try:
            return self.default_nx_data.signal
        except:
            return BAD_NEXUS_FILE

    def _parse_default_axis_name(self) -> str:
        """
        Returns the name of the default axis.
        """
        # pylint: disable=bare-except
        try:
            return self.nx_entry[self.nx_entry.default].axes
        except:
            return BAD_NEXUS_FILE

    def _parse_default_nx_data_name(self):
        """
        Returns the name of the default nxdata.
        """
        # pylint: disable=bare-except
        try:
            return self.nx_entry.default
        except:
            return BAD_NEXUS_FILE

    def _parse_default_nx_data(self) -> np.ndarray:
        """
        Returns the default NXdata.
        """
        # pylint: disable=bare-except
        try:
            return self.nx_entry[self.default_nx_data_name]
        except:
            return BAD_NEXUS_FILE

    @abstractmethod
    def _parse_motors(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary taking the form:

            motor_name: motor_values

        where motor_values is an array containing the value of the motor with
        name motor_name at every point in the scan (even if the motor's value
        is unchanging).
        """


class I07Nexus(NexusBase):
    """
    This class extends NexusBase with methods useful for scraping information
    from nexus files produced at the I07 beamline at Diamond.

    Attrs:
        local_path:
            The local path to the .nxs (metadata) file.
        local_data_path:
            The local path to the data.
        detector_distance:
            The distance between the sample and the detector, or between the
            slit and the detector in DCD geometry.
        setup:
            The scattering geometry. Can be "horizontal", "vertical" or "DCD".
            This will default to "vertical" if the experiment is detected to
            have taken place in EH2.
        local_data_path:
            The local path to data pointed at by this file. Defaults to ''.
            This is particularly important when the data file points to image
            data.
        using_dps:
            A boolean representing whether or not the detector positioning
            system (usually called the DPS system) was in use. This cannot be
            found from the nexus file, since often DPS values will be recorded
            even when it isn't in use.
    """
    # Detectors.
    excalibur_detector_2021 = "excroi"
    excalibur_04_2022 = "exr"
    excalibur_2022_fscan = "EXCALIBUR"
    pilatus_2021 = "pil2roi"
    pilatus_2022 = "PILATUS"
    pilatus_2_stats = "pil2stats"
    pilatus_eh2_2022 = "pil3roi"
    pilatus_eh2_stats = "pil3stats"
    pilatus_eh2_scan = "p3r"
    p2r = "p2r"
    excalibur_08_2023_stats="excstats"
    excalibur_08_2023_roi="excroi"

    # Setups.
    horizontal = "horizontal"
    vertical = "vertical"
    dcd = "DCD"

    def __init__(self,
                 local_path: Union[str, Path],
                 local_data_path: Union[str, Path] = '',
                 detector_distance=None,
                 setup: str = 'horizontal',
                 locate_local_data=True,
                 using_dps=False,
                 experimental_hutch=0):
        # We can store whether we're using the dps system right away.
        self.using_dps = using_dps

        # We need to know what detector we're using before doing any further
        # initialization.
        self.nxfile = nxload(local_path)
        self.nx_entry = self._parse_nx_entry()
        self.detector_name = self._parse_detector_name()

        # Initially, just set the detector rotation to be 0. This will be parsed
        # properly later, but some value is needed to run super().__init__()
        self.det_rot = 0

        # Now we can call super().__init__
        super().__init__(local_path, local_data_path, locate_local_data)

        # Work out which experimental hutch this was carried out in.
        if experimental_hutch==1:
            self.is_eh1=True
            self.is_eh2=False
        elif experimental_hutch==2:
            self.is_eh1=False
            self.is_eh2=True
        else:
            self.is_eh1 = self._is_eh1
            self.is_eh2 = self._is_eh2
            self._check_hutch_parsing()

        # Record the scattering geometry.
        if self.is_eh2:
            self.setup = I07Nexus.vertical
        if self.is_eh1:
            self.setup = setup

        # Now correct our value of det_rot, image_shape and polarisation.
        self.det_rot = self._parse_detector_rot()
        self.image_shape = self._parse_image_shape()
        # The beam is always polarised along the synchrotron x-axis in I07.
        self.polarisation = Polarisation(
            Polarisation.linear,
            Vector3(np.array([1, 0, 0]), Frame(Frame.lab)))

        # Parse the various i07-specific stuff.
        self.detector_distance = detector_distance
        self.transmission = self._parse_transmission()
        self.dcd_circle_radius = self._parse_dcd_circle_radius()
        self.dcd_omega = self._parse_dcd_omega()
        self.delta = self._parse_delta()
        self.gamma = self._parse_gamma()
        self.omega = self._parse_omega()
        self.theta = self._parse_theta()
        self.alpha = self._parse_alpha()
        self.chi = self._parse_chi()
        self.dpsx = self._parse_dpsx()
        self.dpsy = self._parse_dpsy()
        self.dpsz = self._parse_dpsz()
        self.dpsz2 = self._parse_dpsz2()

        # Get the UB and U matrices, if they have been stored.
        self.ub_matrix = self._parse_ub()
        self.u_matrix = self._parse_u()

        # ROIs currently only implemented for the excalibur detector.
        if self.is_excalibur:
            self.signal_regions = self._parse_signal_regions()

        # Work out which images that have been collected should be ignored.
        # This can happen, for example, if the attenuation filters are moving.
        self.ignore_images = []
        filters_moving_frames = self._parse_attenuation_filters_moving()
        if filters_moving_frames is not None:
            self.ignore_images.extend(filters_moving_frames)

    def update_metadata(self, new_metadata: dict):
        """
        Updates various essential bits of metadata that will change this nexus
        file to represent another in what would be a series of scans.
        """
        for metadata_key in new_metadata:
            setattr(self, metadata_key, new_metadata[metadata_key])

    def get_metadata(self):
        """
        Returns various essential bits of metadata that will be needed to
        specify this scan amongst other similar scans.
        """
        meta_dict = {
            'probe_energy': self.probe_energy,
            'local_path': self.local_path,
            'local_data_path': self.local_data_path,
            'dcd_circle_radius': self.dcd_circle_radius,
            'scan_length': self.scan_length,
            'has_hdf5_data': self.has_hdf5_data,
        }
        if self.has_image_data:
            if self.has_hdf5_data:
                meta_dict['hdf5_internal_path'] = self.hdf5_internal_path
                meta_dict['raw_hdf5_path'] = self.raw_hdf5_path
                meta_dict['local_hdf5_path'] = self.local_hdf5_path
            else:
                meta_dict['raw_image_paths'] = self.raw_image_paths
                meta_dict['local_image_paths'] = self.local_image_paths

        return meta_dict

    def update_motors(self, new_motors: Dict[str, np.ndarray]):
        """
        Updates the motor positions of this I07Nexus object. Takes as an
        argument a dictionary of new motor positions that takes the form:

            {'theta': [1,2,3...], 'omega': [4,4,4...], ...}
        """
        for motor in new_motors:
            setattr(self, motor, new_motors[motor])

    def get_motors(self):
        """Returns all the motor positions (for use with update_motors)."""
        return {
            'transmission': self.transmission,
            'dcd_omega': self.dcd_omega,
            'delta': self.delta,
            'gamma': self.gamma,
            'omega': self.omega,
            'theta': self.theta,
            'alpha': self.alpha,
            'chi': self.chi,
            'dpsx': self.dpsx,
            'dpsy': self.dpsy,
            'dpsz': self.dpsz,
            'dpsz2': self.dpsz2
        }

    def get_image(self, image_number: int) -> np.ndarray:
        img = super().get_image(image_number)

        # The following used to be used to deal with broken frames in the
        # excalibur detector, but no longer works. This is kept explicitly in
        # case it becomes useful again in the future.

        # Deal with the fact that the excalibur detector always puts out some
        # bad frames.
        # if self.is_excalibur and self.has_hdf5_data:
        #     bad_frames = [273, 274, 761, 762, 924, 925, 598, 599, 1087, 1088,
        #                   110, 111]
        #     if image_number in bad_frames:
        #         return np.zeros_like(img)

        return img

    def populate_data_from_dat(self, path_to_dat: Union[str, Path]) -> None:
        """
        Overrides the current values of delta, gamma, omega, theta, alpha and
        chi with ones recorded in a .dat file.
        """
        # First use the .dat parsing function to grab a dataframe.
        data_frame = i07_data_from_dat(path_to_dat)

        if self.is_eh1:
            self.delta = data_frame["diff1delta"].to_numpy()
            self.gamma = data_frame["diff1gamma"].to_numpy()
            try:
                self.omega = data_frame["diff1omega"].to_numpy()
            except KeyError:
                self.omega = np.zeros_like(self.delta)
            self.theta = data_frame["diff1theta"].to_numpy()
            try:
                self.alpha = data_frame["diff1alpha"].to_numpy()
            except KeyError:
                self.alpha = np.zeros_like(self.delta)
            self.chi = data_frame["diff1chi"].to_numpy()
        if self.is_eh2:
            self.delta = data_frame["diff2delta"].to_numpy()
            self.gamma = data_frame["diff2gamma"].to_numpy()
            self.omega = data_frame["diff2omega"].to_numpy()
            self.alpha = data_frame["diff2alpha"].to_numpy()

    @property
    def _is_eh1(self) -> bool:
        """
        Works out if the experiment was carried out in experimental hutch 1
        (eh1). Returns the corresponding boolean.

        This information should really be written in the .nxs file, but since
        it isn't we need this kind of hacky workaround.
        """
        # This check is very basic, but at the same time, should be robust. If
        # you're scanning something in ehN, you're bound to have at least one
        # scan field starting with 'diffN' because of how everything is named.
        for field in self.scan_fields:
            # This is not redundant. Users could forget to remove diff2 fields
            # but prepend diff1 to their scan fields. In this case, we should
            # return the truthiness of the first scan field we come across.
            if field.startswith('diff2'):
                return False
            if field.startswith('diff1'):
                return True

        # As of 09/2022, scan_fields may not be populated, in which case it's
        # an empty list in this code. Instead lets look for keys in the detector
        # that begin with diff1.
        # TODO: if this ever breaks, get angry.
        for key in self.nx_entry[self.detector_name]:
            # Make sure that the key is utf-8.
            key = _get_utf_8(key)
            if key.startswith('diff1'):
                return True

        # After digging through IOC's, we saw that if the detector name starts
        # with pil2 it's most likely the P2M, which would be very difficult to
        # use in EH2. So, let's run a check against this too.
        if self.detector_name.startswith("pil2"):
            return True
        if self.detector_name.startswith("p2"):
            return True

        # Similarly, it's very likely that the experiment is in EH1 if the
        # excalibur detector is being used.
        if self.is_excalibur:
            return True

        return False

    @property
    def _is_eh2(self) -> bool:
        """
        Works out if the experiment was carried out in experimental hutch 2
        (eh2). Returns the corresponding boolean.
        """
        # Currently working on the assumption that 'pil3' is associated with
        # the p100k; also assuming that the p100k is only used in eh2. Hardly
        # bulletproof.
        if self._parse_detector_name() in [I07Nexus.pilatus_eh2_2022,
                                           I07Nexus.pilatus_eh2_stats,
                                           I07Nexus.pilatus_eh2_scan]:
            return True

        # This check is very basic, but at the same time, should be robust. If
        # you're scanning something in ehN, you're bound to have at least one
        # scan field starting with 'diffN' because of how everything is named.
        for field in self.scan_fields:
            # This is not redundant. Users could forget to remove diff2 fields
            # but prepend diff1 to their scan fields. In this case, we should
            # return the truthiness of the first scan field we come across.
            if field.startswith('diff1'):
                return False
            if field.startswith('diff2'):
                return True

        # As of 09/2022, scan_fields may not be populated, in which case it's
        # an empty list in this code. Instead lets look for keys in the detector
        # that begin with diff2.
        # TODO: if this ever breaks, get angry.
        for key in self.nx_entry[self.detector_name]:
            # Make sure that the key is utf-8.
            key = _get_utf_8(key)
            if key.startswith('diff2'):
                return True

        return False

    @property
    def is_rotated(self) -> bool:
        """
        Returns True if the detector has been rotated by 90 degrees.
        """
        return (self.det_rot > 89) and (self.det_rot < 91)

    @property
    def has_image_data(self) -> bool:
        """
        It would be pretty weird to not have image data on an i07 nexus file.
        """
        return True

    def _parse_has_hdf5_data(self) -> bool:
        """
        Currently seems like a reasonable way of determining this.
        """
        # If something goes seriously wrong while checking if the file has hdf5
        # data, it probably doesn't! So, we use a broad except in this case.
        # pylint: disable=broad-except
        try:
            # Try to see if our detector's data points at an h5 file.
            if isinstance(self.nx_detector["data"], nx.NXlink):
                if self.nx_detector["data"]._filename.endswith('.h5') or \
                        self.nx_detector["data"]._filename.endswith('.hdf5'):
                    return True
        except Exception:
            # If something went really wrong, there mustn't be .h5 data.
            return False
        return False

    def _check_hutch_parsing(self) -> None:
        """
        Makes sure that the hutch was parsed correctly. If not, raises an error.

        Raises:
            BadNexusFileError
        """
        if self.is_eh1 and self.is_eh2:
            raise BadNexusFileError(
                "This data seemed to belong to both eh1 and eh2.")
        if not (self.is_eh1 or self.is_eh2):
            raise BadNexusFileError(
                "This nexus file didn't seem to belong to eh1 or eh2.")

    @warn_missing_metadata
    def _parse_attenuation_filters_moving(self):
        """
        Attempts to parse whether attenuation filters are moving.
        Note that this is FRAGILE. This will break if either of
        EXCALIBUR_transmission
        or
        attenuation_filters_moving
        change name.
        """
        filters_moving = self.nx_entry[
            "EXCALIBUR_transmission/attenuation_filters_moving"].nxdata
        ignore_images = [
            x for x, num in enumerate(filters_moving) if num == 1
        ]
        return ignore_images

    @warn_missing_metadata
    def _parse_hdf5_internal_path(self) -> str:
        """
        This needs to be implemented properly, as i07 scans *can* have data
        stored in .h5 files.
        """
        return self.nx_detector["data"]._target

    @warn_missing_metadata
    def _parse_raw_hdf5_path(self) -> Union[str, Path]:
        """
        This needs to be implemented properly, as i07 scans *can* have data
        stored in .h5 files.
        """
        filename = self.nx_detector["data"]._filename
        if filename is None:
            filename = self.nx_detector["data"].nxdata
        return filename

    @warn_missing_metadata
    def _parse_probe_energy(self):
        """
        Returns the energy of the probe particle parsed from this NexusFile.
        """
        return float(self.nx_instrument.dcm1energy.value)*1e3

    def _parse_pixel_size(self) -> float:
        """
        Returns the side length of pixels in the detector that's being used.
        """
        if self.is_excalibur:
            return 55e-6
        if self.is_pilatus:
            return 172e-6
        raise ValueError(f"Detector name {self.detector_name} is unknown.")

    def _parse_image_shape(self) -> float:
        """
        Returns the shape of the images we expect to be recorded by this
        detector.
        """
        # In hutch 2, currently only the pilatus 100k is used.
        if self._is_eh2:
            if self.is_rotated:
                return (195, 487)
            return (195, 487)

        # In experimental hutch 1, it could be the P2M or the excalibur.
        if self.is_excalibur:
            #this might not be needed, as rotations can be done elsewhere in code
            if self.is_rotated:
                return 2069, 515
            return 515, 2069
        if self.is_pilatus:
            if self.is_rotated:
                return 1475, 1679
            return 1679, 1475
        raise ValueError(f"Detector name {self.detector_name} is unknown.")

    def _parse_raw_image_paths(self):
        """
        Returns the raw path to the data file. This is useless if you aren't on
        site, but used to guess where you've stored the data file locally.
        """
        if self.is_pilatus:
            path_array = self.nx_detector["image_data"].nxdata
        if self.is_excalibur:
            path_array = [
                self.nx_instrument["excalibur_h5_data/exc_path"].nxdata]
        if len(np.shape(path_array))==1:
            return [x.decode('utf-8') for x in path_array]
        else:
            return [x.decode('utf-8') for listarr in path_array for x in listarr]


    def _parse_nx_detector(self):
        """
        This override is necessary because some i07 .nxs files have multiple
        NXdetectors in their nexus files. What we really want is the appropriate
        camera, which we can parse exploiting the fact that we work out what
        the detector name is elsewhere.
        """
        return self.nx_instrument[self.detector_name]

    def _parse_motors(self) -> Dict[str, np.ndarray]:
        """
        A dictionary of all of the motor positions. This is only useful if you
        know some diffractometer specific keys, so it's kept private to
        encourage users to directly access the cleaner theta, two_theta etc.
        properties.
        """
        motor_names = [
            "diff1delta", "diff1gamma", "diff1omega",  # Basic motors.
            "diff1theta", "diff1chi",  # Basic motors.
            "dcdomega", "dcdc2rad", "diff1prot",  # DCD values.
            "dpsx", "dpsy", "dpsz", "dpsz2"  # DPS values.
        ]

        # For some reason, in at least one nexus file I've seen these names.
        # These are kept in case they are needed for a dodgy nexus file in
        # the future.
        # pylint: disable=unused-variable
        motor_names_eh2_fourc = [
            "fourc.diff2delta", "fourc.diff2gamma",  # Basic motors.
            "fourc.diff2omega", "fourc.diff2alpha"  # Basic motors.
        ]

        # The motors of interest in eh2.
        motor_names_eh2 = [
            "diff2delta", "diff2gamma",  # Basic motors.
            "diff2omega", "diff2alpha"  # Basic motors.
        ]

        # Correct the motor names if we're in experimental hutch 2.
        if self._is_eh2:
            motor_names = motor_names_eh2

        # Set the fourc names if our detector name is pil3roi.
        fourcnames=[I07Nexus.pilatus_eh2_2022,I07Nexus.pilatus_eh2_scan]
        if self.detector_name in fourcnames :
            motor_names = motor_names_eh2_fourc

        motors_dict = {}
        ones = np.ones(self.scan_length)
        for name in motor_names:
            # This could be a link to the data, a single value or a numpy array
            # containing varying values. We need to handle all three cases. The
            # last two cases are handled by multiplying by an array of ones.
            if "value_set" in dir(self.nx_instrument[name]):
                motors_dict[name] = \
                    self.nx_instrument[name].value_set.nxlink.nxdata*ones
                if motors_dict[name] is None:
                    motors_dict[name] = \
                        self.nx_instrument[name].value_set.nxlink.nxdata*ones
            elif "value" in dir(self.nx_instrument[name]):
                motors_dict[name] = self.nx_instrument[name].value.nxdata*ones
        return motors_dict

    @warn_missing_metadata
    def _parse_transmission(self):
        """
        Proportional to the fraction of probe particles allowed by an attenuator
        to strike the sample.
        """
        if "filterset" in self.nx_instrument:
            return float(self.nx_instrument.filterset.transmission)
        elif "fatt" in self.nx_instrument:
            return self.nx_instrument.fatt.transmission.nxdata
        #need to account for instances where there is no transmission data within the nexus 
        # file, set to a default value of 1. 
        else:
            print("No transmission value found, therefore setting to value 1.")
            return 1
        #raise nx.NeXusError("No transmission coefficient found.")

    def _parse_dcd_circle_radius(self) -> float:
        """
        Returns the radius of the DCD circle.
        """
        if self.is_eh1:
            return self.motors["dcdc2rad"][0]

    def _parse_dcd_omega(self) -> np.ndarray:
        """
        Returns a numpy array of the dcd_omega values throughout the scan.
        """
        if self.is_eh1:
            return self.motors["dcdomega"][0]

    def _parse_delta(self) -> np.ndarray:
        """
        Returns a numpy array of the delta values throughout the scan.
        """
        # Force set this to zero if we're using the DPS system. This is
        # important, because moving the diffractometer arm out of the way for
        # dps experiments means that this often ends up at around 90 degrees!
        if self.using_dps:
            return np.zeros((self.scan_length,))
        #also need to set to zero if using p2m without dps
        p2mlist=['pil2stats','pil2roi']
        if self.detector_name in p2mlist:
            return np.zeros((self.scan_length,))

        if self.is_eh2:
            try:
                return self.motors["diff2delta"]
            except KeyError:
                return self.motors["fourc.diff2delta"]
        return self.motors["diff1delta"]

    def _parse_gamma(self) -> np.ndarray:
        """
        Returns a numpy array of the gamma values throughout the scan.
        """
        # Force set this to zero if we're using the DPS system. This is
        # important, because moving the diffractometer arm out of the way for
        # dps experiments means that this could take any value!
        if self.using_dps:
            return np.zeros((self.scan_length,))
        #also need to set to zero if using p2m without dps
        p2mlist=['pil2stats','pil2roi']
        if self.detector_name in p2mlist:
            return np.zeros((self.scan_length,))

        if self.is_eh2:
            try:
                return self.motors["diff2gamma"]
            except KeyError:
                return self.motors["fourc.diff2gamma"]
        return self.motors["diff1gamma"]

    def _parse_omega(self) -> np.ndarray:
        """
        Returns a numpy array of the omega values throughout the scan.
        """
        if self.is_eh2:
            try:
                return self.motors["diff2omega"]
            except KeyError:
                return self.motors["fourc.diff2omega"]
        return self.motors["diff1omega"]

    def _parse_alpha(self) -> np.ndarray:
        """
        Returns a numpy array of the alpha values throughout the scan.
        """
        if self.is_eh2:
            try:
                return self.motors["diff2alpha"]
            except KeyError:
                return self.motors["fourc.diff2alpha"]
        try:
            return self.motors["diff1alpha"]
        except KeyError:
            return np.zeros((self.scan_length))

    def _parse_theta(self) -> np.ndarray:
        """
        Returns a numpy array of the theta values throughout the scan.
        """
        if self.is_eh1:
            return self.motors["diff1theta"]

        # In eh2, just return a bunch of zeros. In reality, there isn't a
        # diff2theta field, but we can equivalently represent that by an array
        # of zeroes.
        return np.zeros((self.scan_length,))

    def _parse_chi(self) -> np.ndarray:
        """
        Returns a numpy array of the chi values throughout the scan.
        """
        if self.is_eh1:
            return self.motors["diff1chi"]

        # In eh2, just return a bunch of zeros. In reality, there isn't a
        # diff2chi field, but we can equivalently represent that by an array
        # of zeroes.
        return np.zeros((self.scan_length,))

    def _parse_detector_rot(self) -> float:
        """
        Returns the orientation of the detector.
        """
        if self.is_eh1:
            return self.motors["diff1prot"][0]
        # For now, assume unrotated detectors in eh2.
        return 0

    def _parse_dpsx(self) -> np.ndarray:
        """
        Returns the x-value of the DPS system. Division by 1e3 converts to m.
        """
        if self.is_eh1:
            return self.motors["dpsx"]/1e3

    def _parse_dpsy(self) -> np.ndarray:
        """
        Returns the y-value of the DPS system. Division by 1e3 converts to m.
        """
        if self.is_eh1:
            return self.motors["dpsy"]/1e3

    def _parse_dpsz(self) -> np.ndarray:
        """
        Returns the z-value of the DPS system. Division by 1e3 converts to m.
        """
        if self.is_eh1:
            return self.motors["dpsz"]/1e3

    def _parse_dpsz2(self) -> np.ndarray:
        """
        Returns the z2-value of the DPS system. Division by 1e3 converts to m.
        """
        if self.is_eh1:
            return self.motors["dpsz2"]/1e3

    def _parse_detector_name(self) -> str:
        """
        Returns the name of the detector that we're using. Because life sucks,
        this is a function of time.
        """
        if "excroi" in self.nx_entry:
            return I07Nexus.excalibur_detector_2021
        if "exr" in self.nx_entry:
            return I07Nexus.excalibur_04_2022
        if "pil2roi" in self.nx_entry:
            return I07Nexus.pilatus_2021
        if "PILATUS" in self.nx_entry:
            return I07Nexus.pilatus_2022
        if "pil2stats" in self.nx_entry:
            return I07Nexus.pilatus_2_stats
        if "p2r" in self.nx_entry:
            return I07Nexus.p2r
        if "EXCALIBUR" in self.nx_entry:
            return I07Nexus.excalibur_2022_fscan
        if "pil3roi" in self.nx_entry:
            return I07Nexus.pilatus_eh2_2022
        if "pil3stats" in self.nx_entry:
            return I07Nexus.pilatus_eh2_stats
        if "p3r" in self.nx_entry:
            return I07Nexus.pilatus_eh2_scan
        if "excstats" in self.nx_entry:
            return I07Nexus.excalibur_08_2023_stats
        if "excroi" in self.nx_entry:
            return I07Nexus.excalibur_08_2023_roi
        

        # If execution reached here, then the entry is (for some reason) missing
        # the detector name. Lets check all the NXdetectors and hope that one of
        # them has a known name. Note that these should *NOT* be combined with
        # the previous if statements. Entry should be checked first, then the
        # NXinstrument.
        if "excroi" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.excalibur_detector_2021
        if "exr" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.excalibur_04_2022
        if "pil2roi" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.pilatus_2021
        if "PILATUS" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.pilatus_2022
        if "pil2stats" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.pilatus_2_stats
        if "p2r" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.p2r
        if "EXCALIBUR" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.excalibur_2022_fscan
        if "pil3roi" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.pilatus_eh2_2022
        if "pil3stats" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.pilatus_eh2_stats
        if "p3r" in self.nx_entry.NXinstrument[0]:
            return I07Nexus.pilatus_eh2_scan

        # pylint: disable=invalid-name

        class GOD_DAMNIT_FIX_YOUR_NXDETECTOR_Error(Exception):
            """
            Extra special exception to raise when the detector name changes.
            """

        # Couldn't recognise the detector.
        raise GOD_DAMNIT_FIX_YOUR_NXDETECTOR_Error(
            "Your detector changed name again...")

    @warn_missing_metadata
    def _parse_signal_regions(self) -> List[Region]:
        """
        Returns a list of region objects that define the location of the signal.
        Currently there is nothing better to do than assume that this is a list
        of length 1.
        """
        # This handles the fact that ROIs used to be stored in a completely
        # different way.
        if self.detector_name == I07Nexus.excalibur_detector_2021:
            return [self._get_ith_region(i=1)]
        # This attempts to parse an invalid json file.
        if self.detector_name == I07Nexus.excalibur_04_2022:
            # Make sure our code executes for bytes and strings.
            try:
                json_str = self.nx_instrument[
                    "ex_rois/excalibur_ROIs"]._value.decode("utf-8")
            except AttributeError:
                json_str = self.nx_instrument["ex_rois/excalibur_ROIs"]._value
            # This is badly formatted and cant be loaded by the json lib. We
            # need to make a series of modifications.
            json_str = json_str.replace('u', '')
            json_str = json_str.replace("'", '"')

            roi_dict = json.loads(json_str)
            return [Region.from_dict(roi_dict['Region_1'])]
        
        if self.detector_name==I07Nexus.excalibur_08_2023_roi:
            regionsfull=list(filter( lambda x: 'Region' in x ,self.nx_instrument.excroi.keys()))
            regionsnum=len(regionsfull)/10
            total_dict={}
            data=self.nx_instrument.excroi
            #create whole dictionary based on full list of regions, but select first value in from X,Y,Width,Height lists
            for n in np.arange(int(regionsnum)):
                roi_dict={f"Region_{n+1}": {"x": data[f'Region_{n+1}_X'][0]._value, "width": data[f'Region_{n+1}_Width'][0]._value, "y": data[f'Region_{n+1}_Y'][0]._value, "height": data[f'Region_{n+1}_Height'][0]._value}}
                total_dict.update(roi_dict)
            #use similar setting to other version where it returns just the region of region1 
            return [Region.from_dict(total_dict['Region_1'])]
        if self.detector_name == I07Nexus.excalibur_2022_fscan:
            # Just ignore the region of interest for fscans.
            return
        if self.detector_name==I07Nexus.excalibur_08_2023_stats:
            # Just ignore use of regions if using excstats.
            return
        raise NotImplementedError()

    @warn_missing_metadata
    def _get_ith_region(self, i: int):
        """
        Returns the ith region of interest found in the .nxs file.

        Args:
            i:
                The region of interest number to return. This number should
                match the ROI name as found in the .nxs file (generally not 0
                indexed).

        Returns:
            The ith region of interest found in the .nxs file.
        """
        x_1 = self.nx_detector[self._get_region_bounds_key(i, 'x_1')][0]
        x_2 = self.nx_detector[self._get_region_bounds_key(
            i, 'Width')][0] + x_1
        y_1 = self.nx_detector[self._get_region_bounds_key(i, 'y_1')][0]
        y_2 = self.nx_detector[self._get_region_bounds_key(
            i, 'Height')][0] + y_1
        return Region(x_1, x_2, y_1, y_2)

    @property
    def background_regions(self) -> List[Region]:
        """
        Returns a list of region objects that define the location of background.
        Currently we just ignore the zeroth region and call the rest of them
        background regions.
        """
        if self.detector_name == I07Nexus.excalibur_detector_2021:
            return [self._get_ith_region(i)
                    for i in range(2, self._number_of_regions+1)]
        if self.detector_name == I07Nexus.excalibur_04_2022:
            # Make sure our code executes for bytes and strings.
            try:
                json_str = self.nx_instrument[
                    "ex_rois/excalibur_ROIs"]._value.decode("utf-8")
            except AttributeError:
                json_str = self.nx_instrument[
                    "ex_rois/excalibur_ROIs"]._value
            # This is badly formatted and cant be loaded by the json lib. We
            # need to make a series of modifications.
            json_str = json_str.replace('u', '')
            json_str = json_str.replace("'", '"')

            roi_dict = json.loads(json_str)
            bkg_roi_list = list(roi_dict.values())[1:]
            return [Region.from_dict(x) for x in bkg_roi_list]

        raise NotImplementedError()

    @property
    def _region_keys(self) -> List[str]:
        """
        Parses all of the detector's dictionary keys and returns all keys
        relating to regions of interest.
        """
        return [key for key in self.nx_detector.keys()
                if key.startswith("Region")]

    @property
    def _number_of_regions(self) -> int:
        """
        Returns the number of regions of interest described by this nexus file.
        This *assumes* that the region keys take the form f'region_{an_int}'.
        """
        split_keys = [key.split('_') for key in self._region_keys]

        return max([int(split_key[1]) for split_key in split_keys])

    def _get_region_bounds_key(self, region_no: int, kind: str) -> List[str]:
        """
        Returns the detector key relating to the bounds of the region of
        interest corresponding to region_no.

        Args:
            region_no:
                An integer corresponding the the particular region of interest
                we're interested in generating a key for.
            kind:
                The kind of region bounds keys we're interested in. This can
                take the values:
                    'x_1', 'width', 'y_1', 'height'
                where '1' can be replaced with 'start' and with/without caps on
                first letter of width/height.

        Raises:
            ValueError if 'kind' argument is not one of the above.

        Returns:
            A list of region bounds keys that is ordered by region number.
        """
        # Note that the x, y swapping is a quirk of the nexus standard, and is
        # related to which axis on the detector varies most rapidly in memory.
        if kind in ('x_1', 'x_start'):
            insert = 'X'
        elif kind in ('width', 'Width'):
            insert = 'Width'
        elif kind in ('y_1', 'y_start'):
            insert = 'Y'
        elif kind in ('height', 'Height'):
            insert = 'Height'
        else:
            raise ValueError("Didn't recognise 'kind' argument.")

        return f"Region_{region_no}_{insert}"

    @property
    def is_excalibur(self) -> bool:
        """
        Returns whether or not we're currently using the excalibur detector.
        """
        return self.detector_name in ['excroi', 'exr', 'EXCALIBUR','excstats']

    @property
    def is_pilatus(self) -> bool:
        """
        Returns whether or not we're currently using the pilatus detector.
        """
        return self.detector_name in [I07Nexus.pilatus_2021,
                                      I07Nexus.pilatus_2022,
                                      I07Nexus.pilatus_2_stats,
                                      I07Nexus.p2r,
                                      I07Nexus.pilatus_eh2_2022,
                                      I07Nexus.pilatus_eh2_stats,
                                      I07Nexus.pilatus_eh2_scan]

    @warn_missing_metadata
    def _parse_u(self) -> np.ndarray:
        """
        Parses the UB matrix from a .nxs file, if it has been stored. If it
        hasn't, returns None.
        """
        # This may result in some warnings when reading older data.
        return self.nx_instrument["diffcalchdr.diffcalc_u"].value.nxdata

    @warn_missing_metadata
    def _parse_ub(self) -> np.ndarray:
        """
        Parses the UB matrix from a .nxs file, if it has been stored. If it
        hasn't, returns None.
        """
        # This may result in some warnings when reading older data.
        return self.nx_instrument["diffcalchdr.diffcalc_ub"].value.nxdata


class I10Nexus(NexusBase):
    """
    This class extends NexusBase with methods useful for scraping information
    from nexus files produced at the I10 beamline at Diamond.
    """

    # We might need to check which instrument we're using at some point.
    rasor_instrument = "rasor"

    def __init__(self,
                 local_path: Union[str, Path],
                 local_data_path: Union[str, Path] = '',
                 detector_distance: float = None,
                 locate_local_data: bool = True):
        super().__init__(local_path, local_data_path, locate_local_data)

        # TODO: properly parse this when this becomes relevant.
        self.polarisation = NotImplemented

        # Warn the user if detector distance hasn't been set.
        if detector_distance is None:
            warn(MissingMetadataWarning(
                "Detector distance has not been set. At I10, sample-detector "
                "distance is not recorded in the nexus file, and must be "
                "input manually when using this library if it is needed."))

        # Initialize the i10 specific stuff.
        self.detector_distance = detector_distance
        self.theta = self._parse_theta()
        self.theta_area = self._parse_theta_area()
        self.two_theta = self._parse_two_theta()
        self.two_theta_area = self._parse_two_theta_area()
        self.chi = self._parse_chi()

    @property
    def has_image_data(self) -> bool:
        """For now, assume all i10 data we're given is image data."""
        return True

    def _parse_has_hdf5_data(self) -> bool:
        """As of 31/05/2022, i10 does not output hdf5 data, only .tiffs."""
        return False

    def _parse_hdf5_internal_path(self) -> str:
        """Trivially raises, but we need to implement the abstractmethod"""
        return super()._parse_hdf5_internal_path()

    def _parse_raw_hdf5_path(self) -> Union[str, Path]:
        """Trivially raises, but we need to implement the abstractmethod"""
        return super()._parse_raw_hdf5_path()

    def _parse_raw_image_paths(self) -> List[str]:
        """
        Returns a list of paths to the .tiff images recorded during this scan.
        These are the same paths that were originally recorded during the scan,
        so will point at some directory in the diamond filesystem.
        """
        return [x.decode('utf-8') for x in self.default_signal]

    def _parse_probe_energy(self):
        """
        Returns the energy of the probe particle parsed from this NexusFile.
        """
        return float(self.nx_instrument.pgm.energy)

    def _parse_pixel_size(self) -> float:
        """
        All detectors on I10 have 13.5 micron pixels.
        """
        return 13.5e-6

    def _parse_image_shape(self) -> Tuple[int]:
        """
        Returns the shape of detector images. This is easy in I10, since they're
        both 2048**2 square detectors.
        """
        return 2048, 2048

    def _parse_motors(self) -> Dict[str, np.ndarray]:
        """
        A dictionary of all of the motor positions. This is only useful if you
        know some diffractometer specific keys, so it's kept private to
        encourage users to directly access the cleaner theta, two_theta etc.
        properties.
        """
        instr_motor_names = ["th", "tth", "chi"]
        diff_motor_names = ["theta", "2_theta", "chi"]

        motors_dict = {
            x: np.ones(self.scan_length) *
            self.nx_instrument.rasor.diff[y]._value
            for x, y in zip(instr_motor_names, diff_motor_names)}

        for name in instr_motor_names:
            try:
                motors_dict[name] = self.nx_instrument[name].value._value
            except KeyError:
                pass
        return motors_dict

    def _parse_theta(self) -> np.ndarray:
        """
        Returns the current theta value of the diffractometer, as parsed from
        the nexus file. Note that this will be different to thArea in GDA.
        """
        return self.motors["th"]

    def _parse_two_theta(self) -> np.ndarray:
        """
        Returns the current two-theta value of the diffractometer, as parsed
        from the nexus file. Note that this will be different to tthArea in GDA.
        """
        return self.motors["tth"]

    def _parse_theta_area(self) -> np.ndarray:
        """
        Returns the values of the thArea virtual motor during this scan.
        """
        return 180 - self.theta

    def _parse_two_theta_area(self) -> np.ndarray:
        """
        Returns the values of the tthArea virtual motor during this scan.
        """
        return 90 - self.two_theta

    def _parse_chi(self) -> np.ndarray:
        """
        Returns the current chi value of the diffractometer.
        """
        return 90 - self.motors["chi"]
