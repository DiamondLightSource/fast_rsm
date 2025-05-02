:py:mod:`binning`
=================

.. py:module:: binning

.. autodoc2-docstring:: binning
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_fix_delta_q_geometry <binning._fix_delta_q_geometry>`
     - .. autodoc2-docstring:: binning._fix_delta_q_geometry
          :summary:
   * - :py:obj:`_fix_intensity_geometry <binning._fix_intensity_geometry>`
     - .. autodoc2-docstring:: binning._fix_intensity_geometry
          :summary:
   * - :py:obj:`finite_diff_grid <binning.finite_diff_grid>`
     - .. autodoc2-docstring:: binning.finite_diff_grid
          :summary:
   * - :py:obj:`finite_diff_shape <binning.finite_diff_shape>`
     - .. autodoc2-docstring:: binning.finite_diff_shape
          :summary:
   * - :py:obj:`linear_bin <binning.linear_bin>`
     - .. autodoc2-docstring:: binning.linear_bin
          :summary:
   * - :py:obj:`fast_linear_bin <binning.fast_linear_bin>`
     - .. autodoc2-docstring:: binning.fast_linear_bin
          :summary:
   * - :py:obj:`weighted_bin_3d <binning.weighted_bin_3d>`
     - .. autodoc2-docstring:: binning.weighted_bin_3d
          :summary:
   * - :py:obj:`weighted_bin_1d <binning.weighted_bin_1d>`
     - .. autodoc2-docstring:: binning.weighted_bin_1d
          :summary:
   * - :py:obj:`hist_shape <binning.hist_shape>`
     - .. autodoc2-docstring:: binning.hist_shape
          :summary:
   * - :py:obj:`linear_bin_histdd <binning.linear_bin_histdd>`
     - .. autodoc2-docstring:: binning.linear_bin_histdd
          :summary:

API
~~~

.. py:function:: _fix_delta_q_geometry(arr: numpy.ndarray) -> numpy.ndarray
   :canonical: binning._fix_delta_q_geometry

   .. autodoc2-docstring:: binning._fix_delta_q_geometry

.. py:function:: _fix_intensity_geometry(arr: numpy.ndarray) -> numpy.ndarray
   :canonical: binning._fix_intensity_geometry

   .. autodoc2-docstring:: binning._fix_intensity_geometry

.. py:function:: finite_diff_grid(start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray)
   :canonical: binning.finite_diff_grid

   .. autodoc2-docstring:: binning.finite_diff_grid

.. py:function:: finite_diff_shape(start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray)
   :canonical: binning.finite_diff_shape

   .. autodoc2-docstring:: binning.finite_diff_shape

.. py:function:: linear_bin(coords: numpy.ndarray, intensities: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray) -> numpy.ndarray
   :canonical: binning.linear_bin

   .. autodoc2-docstring:: binning.linear_bin

.. py:function:: fast_linear_bin(coords: numpy.ndarray, intensities: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray) -> numpy.ndarray
   :canonical: binning.fast_linear_bin

   .. autodoc2-docstring:: binning.fast_linear_bin

.. py:function:: weighted_bin_3d(coords: numpy.ndarray, weights: numpy.ndarray, out: numpy.ndarray, count: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, min_intensity=None) -> numpy.ndarray
   :canonical: binning.weighted_bin_3d

   .. autodoc2-docstring:: binning.weighted_bin_3d

.. py:function:: weighted_bin_1d(coords: numpy.ndarray, weights: numpy.ndarray, out: numpy.ndarray, count: numpy.ndarray, start: float, stop: float, step: float)
   :canonical: binning.weighted_bin_1d

   .. autodoc2-docstring:: binning.weighted_bin_1d

.. py:function:: hist_shape(start, stop, step)
   :canonical: binning.hist_shape

   .. autodoc2-docstring:: binning.hist_shape

.. py:function:: linear_bin_histdd(coords: numpy.ndarray, intensities: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray) -> numpy.ndarray
   :canonical: binning.linear_bin_histdd

   .. autodoc2-docstring:: binning.linear_bin_histdd
