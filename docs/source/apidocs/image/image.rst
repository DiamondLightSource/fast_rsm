:py:mod:`image`
===============

.. py:module:: image

.. autodoc2-docstring:: image
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Image <image.Image>`
     - .. autodoc2-docstring:: image.Image
          :summary:

API
~~~

.. py:class:: Image(metadata: fast_rsm.rsm_metadata.RSMMetadata, index: int, load_image=True)
   :canonical: image.Image

   .. autodoc2-docstring:: image.Image

   .. rubric:: Initialization

   .. autodoc2-docstring:: image.Image.__init__

   .. py:method:: _correct_img_axes()
      :canonical: image.Image._correct_img_axes

      .. autodoc2-docstring:: image.Image._correct_img_axes

   .. py:method:: generate_mask(min_intensity: typing.Union[float, int]) -> numpy.ndarray
      :canonical: image.Image.generate_mask

      .. autodoc2-docstring:: image.Image.generate_mask

   .. py:method:: add_processing_step(function) -> None
      :canonical: image.Image.add_processing_step

      .. autodoc2-docstring:: image.Image.add_processing_step

   .. py:property:: data
      :canonical: image.Image.data

      .. autodoc2-docstring:: image.Image.data

   .. py:method:: q_vectors(frame: diffraction_utils.Frame, indices: tuple = None, oop='y', lorentz_correction: bool = False, pol_correction: bool = True) -> numpy.ndarray
      :canonical: image.Image.q_vectors

      .. autodoc2-docstring:: image.Image.q_vectors

   .. py:method:: q_vector_array(frame: diffraction_utils.Frame, oop='y', lorentz_correction: bool = False, pol_correction: bool = True) -> numpy.ndarray
      :canonical: image.Image.q_vector_array

      .. autodoc2-docstring:: image.Image.q_vector_array
