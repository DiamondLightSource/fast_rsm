"""
This module contains the DataFileBase class, returned by parser methods in the
islatu.io module. This class provides a consistent way to refer to metadata
returned by different detectors/instruments.
"""

import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
from PIL import Image as PILImageModule

from .polarisation import Polarisation


class NoHdf5Error(Exception):
    """
    Exception to be raised if there's no hdf5 data pointed to by a data file,
    but a user tries to access hdf5-specific information (like an internal data
    path etc.)
    """


class NoImagesError(Exception):
    """
    An exception to be raised if methods relating to image parsing are called
    when the datafile doesn't actually point to image data (i.e. when the
    data file points to data acquired using a point detector).
    """


class DataFileBase(ABC):
    """
    An ABC for classes that store metadata parsed from data files. This defines
    the properties that must be implemented by parsing classes.

    Attrs:
        local_path:
            The local path to *this* file.
        local_data_path:
            The local path to data pointed at by this file. Defaults to ''.
            This is particularly important when the data file points to image
            data.
        raw_image_paths:
            The raw paths to the image data, if there is any.
    """

    def __init__(self,
                 local_path: Union[str, Path],  # The path to this file.
                 local_data_path: Union[str, Path] = '',  # Path to the data.
                 # If True, will raise if local data cannot be found.
                 locate_local_data=True):
        self.local_path = local_path
        self.local_data_path = local_data_path

        # Default the beam to be unpolarised.
        self.polarisation = Polarisation(Polarisation.unpolarised)

        # Run the various parsers to initialize non-trivial attributes.
        self.probe_energy = self._parse_probe_energy()
        self.default_axis_name = self._parse_default_axis_name()
        self.default_signal_name = self._parse_default_signal_name()
        self.default_axis = self._parse_default_axis()
        self.default_signal = self._parse_default_signal()
        self.scan_length = self._parse_scan_length()
        self.has_hdf5_data = self._parse_has_hdf5_data()

        # Do the image specific initialization.
        if self.has_image_data:
            self.image_shape = self._parse_image_shape()
            self.pixel_size = self._parse_pixel_size()
            # Image data is either all stored in 1 big hdf5 file, or as separate
            # images.
            if self.has_hdf5_data:
                self.hdf5_internal_path = self._parse_hdf5_internal_path()
                self.raw_hdf5_path = self._parse_raw_hdf5_path()
                if locate_local_data:
                    self.local_hdf5_path = self._parse_local_hdf5_path()
            else:
                self.raw_image_paths = self._parse_raw_image_paths()
                if locate_local_data:
                    self.local_image_paths = self._parse_local_image_paths()

    @abstractmethod
    def _parse_pixel_size(self):
        """
        Returns the size of the pixels in the images associated with this
        DataFile, if there are any.
        """
        if not self.has_image_data:
            raise NoImagesError()

    @abstractmethod
    def _parse_image_shape(self):
        """
        Returns the shape of image data, if there is any.
        """
        if not self.has_image_data:
            raise NoImagesError()

    @abstractmethod
    def _parse_probe_energy(self):
        """
        The energy of the probe particle, in eV.
        """

    @abstractmethod
    def _parse_default_signal(self) -> np.ndarray:
        """
        Returns the default signal, where "default signal" should be understood
        in the NeXus sense to mean the experiment's default independent
        variable.
        """

    @abstractmethod
    def _parse_default_signal_name(self) -> str:
        """
        The name of the default signal axis, e.g. "intensity" or "roi_1_sum" or
        "image_paths" or some other such appropriate string.
        """

    @abstractmethod
    def _parse_default_axis(self) -> np.ndarray:
        """
        Returns a numpy array of data associated with the default axis, where
        "default axis" should be understood in the NeXus sense to mean the
        experiment's dependent variable.
        """
        raise NotImplementedError()

    @abstractmethod
    def _parse_default_axis_name(self) -> str:
        """
        Returns the name of the default axis, as it was recorded in the data
        file stored at local_path.
        """
        raise NotImplementedError()

    def _parse_scan_length(self) -> int:
        """
        Returns the number of data points collected during this scan.
        """
        return np.size(self.default_signal)

    @abstractmethod
    def _parse_hdf5_internal_path(self) -> str:
        """
        Returns the internal data path to hdf5 data, if instances of this data
        file have hdf5 data. If not, raises NoHdf5Error.
        """
        if not self.has_hdf5_data:
            raise NoHdf5Error()

    @abstractmethod
    def _parse_has_hdf5_data(self) -> bool:
        """
        Returns whether or not this data file points to some hdf5 data.
        """

    @property
    @abstractmethod
    def has_image_data(self) -> bool:
        """
        Returns whether or not this data file points to image data (the
        alternative would be data acquired using a point detector).
        """

    def get_image(self, image_number: int) -> np.ndarray:
        """
        Returns the numpy array representation of one of the images in the scan
        represented by this nexus file.

        Args:
            image_number:
                The index of the image you would like to load.

        Returns:
            A numpy array representation of the image of interest.
        """
        if not self.has_image_data:
            # Clearly, this method shouldn't have been called.
            raise NoImagesError()

        if self.has_hdf5_data:
            # If this is hdf5 data, open the file and grab the correct image.
            with h5py.File(self.local_hdf5_path, "r") as open_file:
                dataset = open_file[self.hdf5_internal_path]
                img_arr = np.array(dataset[image_number])
                return img_arr
        else:
            # If these are separately stored images, grab the correct path from
            # local_image_paths and load that specific image.
            image_path = self.local_image_paths[image_number]
            return np.array(PILImageModule.open(image_path))

    @abstractmethod
    def _parse_raw_hdf5_path(self) -> Union[str, Path]:
        """
        Returns the path to the hdf5 file pointed at by this data file, if there
        is one. If there isn't, raises NoHdf5Error.

        Raises:
            NoHdf5Error if this DataFile doesn't point at hdf5 data.

        Returns:
            The path to the raw hdf5 files pointed at by this DataFile.
        """
        if not self.has_hdf5_data:
            raise NoHdf5Error()

    def _parse_local_hdf5_path(self) -> Union[str, Path]:
        """
        Returns the local path to the hdf5 file pointed at by the data file, if
        there is one. If there isn't, raises NoHdf5Error.

        Raises:
            NoHdf5Error if this DataFile doesn't point at hdf5 data.

        Returns:
            The path to the local hdf5 files associated with this DataFile.
        """
        if not self.has_hdf5_data:
            raise NoHdf5Error()

        # Note that _try_to_find_files always returns a list of found files.
        # In this case, we expect to find only one h5 file containing all of
        # the images. This syntax not only looks cool, but handily raises if
        # we found more than one hdf5 file with that name.
        hdf5_file, = _try_to_find_files(
            [self.raw_hdf5_path], [self.local_data_path, self.local_path])
        return hdf5_file

    @abstractmethod
    def _parse_raw_image_paths(self) -> List[str]:
        """
        Returns a list of paths to the .tiff images recorded during this scan.
        These are the same paths that were originally recorded during the scan,
        so will point at some directory in the diamond filesystem for diamond
        data files, for example.
        """
        if not self.has_image_data:
            raise NoImagesError()

    def _parse_local_image_paths(self) -> List[str]:
        """
        Returns a list of local image paths. Raises FileNotFoundError if any of
        the images cannot be found. These local paths can be used to directly
        load the images.

        Raises:
            FileNotFoundError if any of the images cant be found.
        """
        if not self.has_image_data:
            raise NoImagesError()

        return _try_to_find_files(self.raw_image_paths,
                                  [self.local_data_path, self.local_path])


def _try_to_find_files(filenames: List[str],
                       additional_search_paths: List[str]):
    """
    Check that data files exist if the file parsed by parser pointed to a
    separate file containing intensity information. If the intensity data
    file could not be found in its original location, check a series of
    probable locations for the data file. If the data file is found in one
    of these locations, update file's entry in self.data.

    Returns:
        :py:attr:`list` of :py:attr:`str`:
            List of the corrected, actual paths to the files.
    """
    found_files = []

    # This function was written to handle strings, not pathlib.Paths.
    # It would be nice to update this one day, but for now I'm just casting
    # Paths to strings.
    filenames = [str(x) for x in filenames if x!=""]
    additional_search_paths = [str(x) for x in additional_search_paths]

    # If we had only one file, make a list out of it.
    if not hasattr(filenames, "__iter__"):
        filenames = [filenames]

    cwd = os.getcwd()
    start_dirs = [
        cwd,  # maybe file is stored near the current working dir
        # To search additional directories, add them in here manually.
    ]
    start_dirs.extend(additional_search_paths)

    local_start_directories = [x.replace('\\', '/') for x in start_dirs]
    num_start_directories = len(local_start_directories)

    # Now extend the additional search paths.
    for i in range(num_start_directories):
        search_path = local_start_directories[i]
        split_srch_path = search_path.split('/')
        for j in range(len(split_srch_path)):
            extra_path_list = split_srch_path[:-(j+1)]
            extra_path = '/'.join(extra_path_list)
            local_start_directories.append(extra_path)
    
    good_local_start_directories=[x for x in local_start_directories if x!='']

    # This line allows for a loading bar to show as we check the file.
    for i, _ in enumerate(filenames):
        # Better to be safe... Note: windows is happy with / even though it
        # defaults to \
        filenames[i] = str(filenames[i]).replace('\\', '/')

        # Maybe we can see the file in its original storage location?
        if os.path.isfile(filenames[i]):
            found_files.append(filenames[i])
            continue

        # If not, maybe it's stored locally? If the file was stored at
        # location /a1/a2/.../aN/file originally, for a local directory LD,
        # check locations LD/aj/aj+1/.../aN for all j<N and all LD's of
        # interest. This algorithm is a generalization of Andrew McCluskey's
        # original approach.

        # now generate a list of all directories that we'd like to check
        candidate_paths = []
        split_file_path = str(filenames[i]).split('/')
        for j in range(len(split_file_path)):
            local_guess = '/'.join(split_file_path[j:])
            for start_dir in good_local_start_directories:
                candidate_paths.append(
                    os.path.join(start_dir, local_guess))

        # Iterate over each of the candidate paths to see if any of them contain
        # the data file we're looking for.
        found_file = False
        for candidate_path in candidate_paths:
            if os.path.isfile(candidate_path):
                # File found - add the correct file location to found_files
                found_files.append(candidate_path)
                found_file = not found_file
                # debug.log("Data file found at " + candidate_path + ".")
                break

        # If we didn't find the file, tell the user.
        if not found_file:
            raise FileNotFoundError(
                f"The data file with the name {filenames[i]} could "
                "not be found. "
                f"Arguments passed were: filenames: {filenames} and "
                f"additional_search_paths: {additional_search_paths}."
            )
    return found_files
