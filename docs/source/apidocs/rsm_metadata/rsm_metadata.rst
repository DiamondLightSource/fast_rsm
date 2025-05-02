:py:mod:`rsm_metadata`
======================

.. py:module:: rsm_metadata

.. autodoc2-docstring:: rsm_metadata
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RSMMetadata <rsm_metadata.RSMMetadata>`
     - .. autodoc2-docstring:: rsm_metadata.RSMMetadata
          :summary:

API
~~~

.. py:class:: RSMMetadata(diffractometer: diffraction_utils.DiffractometerBase, beam_centre: typing.Tuple[int, int], mask_pixels: tuple = None, mask_regions: typing.List[diffraction_utils.Region] = None)
   :canonical: rsm_metadata.RSMMetadata

   .. autodoc2-docstring:: rsm_metadata.RSMMetadata

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsm_metadata.RSMMetadata.__init__

   .. py:method:: _correct_beam_centre()
      :canonical: rsm_metadata.RSMMetadata._correct_beam_centre

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata._correct_beam_centre

   .. py:method:: update_i07_nx(motors: typing.Dict[str, numpy.ndarray], metadata: dict)
      :canonical: rsm_metadata.RSMMetadata.update_i07_nx

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.update_i07_nx

   .. py:method:: get_detector_distance(index: int) -> float
      :canonical: rsm_metadata.RSMMetadata.get_detector_distance

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.get_detector_distance

   .. py:property:: solid_angles
      :canonical: rsm_metadata.RSMMetadata.solid_angles

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.solid_angles

   .. py:property:: vertical_pixel_offsets
      :canonical: rsm_metadata.RSMMetadata.vertical_pixel_offsets

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.vertical_pixel_offsets

   .. py:property:: horizontal_pixel_offsets
      :canonical: rsm_metadata.RSMMetadata.horizontal_pixel_offsets

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.horizontal_pixel_offsets

   .. py:method:: get_vertical_pixel_distances(idx: int) -> numpy.ndarray
      :canonical: rsm_metadata.RSMMetadata.get_vertical_pixel_distances

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.get_vertical_pixel_distances

   .. py:method:: get_horizontal_pixel_distances(index: int) -> numpy.ndarray
      :canonical: rsm_metadata.RSMMetadata.get_horizontal_pixel_distances

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.get_horizontal_pixel_distances

   .. py:property:: relative_polar
      :canonical: rsm_metadata.RSMMetadata.relative_polar

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.relative_polar

   .. py:property:: relative_azimuth
      :canonical: rsm_metadata.RSMMetadata.relative_azimuth

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.relative_azimuth

   .. py:property:: incident_wavelength
      :canonical: rsm_metadata.RSMMetadata.incident_wavelength

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.incident_wavelength

   .. py:property:: k_incident_length
      :canonical: rsm_metadata.RSMMetadata.k_incident_length

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata.k_incident_length

   .. py:method:: _init_solid_angles()
      :canonical: rsm_metadata.RSMMetadata._init_solid_angles

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata._init_solid_angles

   .. py:method:: _init_vertical_pixel_offsets(image_shape: int = None)
      :canonical: rsm_metadata.RSMMetadata._init_vertical_pixel_offsets

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata._init_vertical_pixel_offsets

   .. py:method:: _init_horizontal_pixel_offsets(image_shape: int = None)
      :canonical: rsm_metadata.RSMMetadata._init_horizontal_pixel_offsets

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata._init_horizontal_pixel_offsets

   .. py:method:: _init_relative_polar(image_shape: int = None)
      :canonical: rsm_metadata.RSMMetadata._init_relative_polar

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata._init_relative_polar

   .. py:method:: _init_relative_azimuth(image_shape: int = None)
      :canonical: rsm_metadata.RSMMetadata._init_relative_azimuth

      .. autodoc2-docstring:: rsm_metadata.RSMMetadata._init_relative_azimuth
