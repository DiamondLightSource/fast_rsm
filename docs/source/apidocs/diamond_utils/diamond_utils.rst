:py:mod:`diamond_utils`
=======================

.. py:module:: diamond_utils

.. autodoc2-docstring:: diamond_utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`load_exact_map <diamond_utils.load_exact_map>`
     - .. autodoc2-docstring:: diamond_utils.load_exact_map
          :summary:
   * - :py:obj:`intensity_vs_q_exact <diamond_utils.intensity_vs_q_exact>`
     - .. autodoc2-docstring:: diamond_utils.intensity_vs_q_exact
          :summary:
   * - :py:obj:`qxy_qz_exact <diamond_utils.qxy_qz_exact>`
     - .. autodoc2-docstring:: diamond_utils.qxy_qz_exact
          :summary:
   * - :py:obj:`q_to_theta <diamond_utils.q_to_theta>`
     - .. autodoc2-docstring:: diamond_utils.q_to_theta
          :summary:
   * - :py:obj:`get_volume_and_bounds <diamond_utils.get_volume_and_bounds>`
     - .. autodoc2-docstring:: diamond_utils.get_volume_and_bounds
          :summary:
   * - :py:obj:`_project_to_1d <diamond_utils._project_to_1d>`
     - .. autodoc2-docstring:: diamond_utils._project_to_1d
          :summary:
   * - :py:obj:`intensity_vs_q <diamond_utils.intensity_vs_q>`
     - .. autodoc2-docstring:: diamond_utils.intensity_vs_q
          :summary:
   * - :py:obj:`intensity_vs_tth <diamond_utils.intensity_vs_tth>`
     - .. autodoc2-docstring:: diamond_utils.intensity_vs_tth
          :summary:
   * - :py:obj:`intensity_vs_l <diamond_utils.intensity_vs_l>`
     - .. autodoc2-docstring:: diamond_utils.intensity_vs_l
          :summary:
   * - :py:obj:`save_binoculars_hdf5 <diamond_utils.save_binoculars_hdf5>`
     - .. autodoc2-docstring:: diamond_utils.save_binoculars_hdf5
          :summary:
   * - :py:obj:`most_recent_cluster_output <diamond_utils.most_recent_cluster_output>`
     - .. autodoc2-docstring:: diamond_utils.most_recent_cluster_output
          :summary:
   * - :py:obj:`most_recent_cluster_error <diamond_utils.most_recent_cluster_error>`
     - .. autodoc2-docstring:: diamond_utils.most_recent_cluster_error
          :summary:

API
~~~

.. py:function:: load_exact_map(q_vector_path: str, intensities_path: str) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
   :canonical: diamond_utils.load_exact_map

   .. autodoc2-docstring:: diamond_utils.load_exact_map

.. py:function:: intensity_vs_q_exact(q_vectors: numpy.ndarray, intensities: numpy.ndarray, num_bins=1000) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
   :canonical: diamond_utils.intensity_vs_q_exact

   .. autodoc2-docstring:: diamond_utils.intensity_vs_q_exact

.. py:function:: qxy_qz_exact(q_vectors: str, intensities: str, qxy_bins: int = 1000, qz_bins: int = 1000, qz_axis: int = 2) -> numpy.ndarray
   :canonical: diamond_utils.qxy_qz_exact

   .. autodoc2-docstring:: diamond_utils.qxy_qz_exact

.. py:function:: q_to_theta(q_values: numpy.ndarray, energy: float) -> numpy.ndarray
   :canonical: diamond_utils.q_to_theta

   .. autodoc2-docstring:: diamond_utils.q_to_theta

.. py:function:: get_volume_and_bounds(path_to_npy: str) -> typing.Tuple[numpy.ndarray]
   :canonical: diamond_utils.get_volume_and_bounds

   .. autodoc2-docstring:: diamond_utils.get_volume_and_bounds

.. py:function:: _project_to_1d(volume: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, num_bins: int = 1000, bin_size: float = None, tth=False, only_l=False, energy=None)
   :canonical: diamond_utils._project_to_1d

   .. autodoc2-docstring:: diamond_utils._project_to_1d

.. py:function:: intensity_vs_q(output_file_name: str, volume: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, num_bins: int = 1000, bin_size: float = None)
   :canonical: diamond_utils.intensity_vs_q

   .. autodoc2-docstring:: diamond_utils.intensity_vs_q

.. py:function:: intensity_vs_tth(output_file_name: str, volume: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, energy: float, num_bins: int = 1000, bin_size: float = None)
   :canonical: diamond_utils.intensity_vs_tth

   .. autodoc2-docstring:: diamond_utils.intensity_vs_tth

.. py:function:: intensity_vs_l(output_file_name: str, volume: numpy.ndarray, start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, num_bins: int = 1000, bin_size: float = None)
   :canonical: diamond_utils.intensity_vs_l

   .. autodoc2-docstring:: diamond_utils.intensity_vs_l

.. py:function:: save_binoculars_hdf5(path_to_npy: numpy.ndarray, output_path: str, joblines, pythonlocation, outvars=None)
   :canonical: diamond_utils.save_binoculars_hdf5

   .. autodoc2-docstring:: diamond_utils.save_binoculars_hdf5

.. py:function:: most_recent_cluster_output()
   :canonical: diamond_utils.most_recent_cluster_output

   .. autodoc2-docstring:: diamond_utils.most_recent_cluster_output

.. py:function:: most_recent_cluster_error()
   :canonical: diamond_utils.most_recent_cluster_error

   .. autodoc2-docstring:: diamond_utils.most_recent_cluster_error
