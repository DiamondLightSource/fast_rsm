:py:mod:`scan`
==============

.. py:module:: scan

.. autodoc2-docstring:: scan
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Scan <scan.Scan>`
     - .. autodoc2-docstring:: scan.Scan
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`check_shared_memory <scan.check_shared_memory>`
     - .. autodoc2-docstring:: scan.check_shared_memory
          :summary:
   * - :py:obj:`init_process_pool <scan.init_process_pool>`
     - .. autodoc2-docstring:: scan.init_process_pool
          :summary:
   * - :py:obj:`init_pyfai_process_pool <scan.init_pyfai_process_pool>`
     - .. autodoc2-docstring:: scan.init_pyfai_process_pool
          :summary:
   * - :py:obj:`pyfai_stat_exitangles <scan.pyfai_stat_exitangles>`
     - .. autodoc2-docstring:: scan.pyfai_stat_exitangles
          :summary:
   * - :py:obj:`pyfai_stat_qmap <scan.pyfai_stat_qmap>`
     - .. autodoc2-docstring:: scan.pyfai_stat_qmap
          :summary:
   * - :py:obj:`pyfai_stat_ivsq <scan.pyfai_stat_ivsq>`
     - .. autodoc2-docstring:: scan.pyfai_stat_ivsq
          :summary:
   * - :py:obj:`pyfai_move_exitangles <scan.pyfai_move_exitangles>`
     - .. autodoc2-docstring:: scan.pyfai_move_exitangles
          :summary:
   * - :py:obj:`pyfai_move_qmap <scan.pyfai_move_qmap>`
     - .. autodoc2-docstring:: scan.pyfai_move_qmap
          :summary:
   * - :py:obj:`pyfai_move_ivsq <scan.pyfai_move_ivsq>`
     - .. autodoc2-docstring:: scan.pyfai_move_ivsq
          :summary:
   * - :py:obj:`_on_exit <scan._on_exit>`
     - .. autodoc2-docstring:: scan._on_exit
          :summary:
   * - :py:obj:`chunk <scan.chunk>`
     - .. autodoc2-docstring:: scan.chunk
          :summary:
   * - :py:obj:`_chunk_indices <scan._chunk_indices>`
     - .. autodoc2-docstring:: scan._chunk_indices
          :summary:
   * - :py:obj:`_bin_one_map <scan._bin_one_map>`
     - .. autodoc2-docstring:: scan._bin_one_map
          :summary:
   * - :py:obj:`bin_maps_with_indices <scan.bin_maps_with_indices>`
     - .. autodoc2-docstring:: scan.bin_maps_with_indices
          :summary:

API
~~~

.. py:function:: check_shared_memory(shared_mem_name: str) -> None
   :canonical: scan.check_shared_memory

   .. autodoc2-docstring:: scan.check_shared_memory

.. py:function:: init_process_pool(locks: typing.List[multiprocessing.Lock], num_threads: int, metadata: fast_rsm.rsm_metadata.RSMMetadata, frame: diffraction_utils.Frame, shape: tuple, output_file_name: str = None) -> None
   :canonical: scan.init_process_pool

   .. autodoc2-docstring:: scan.init_process_pool

.. py:function:: init_pyfai_process_pool(locks: typing.List[multiprocessing.Lock], num_threads: int, metadata: fast_rsm.rsm_metadata.RSMMetadata, shapeqi: tuple, shapecake: tuple, shapeqpqpmap: tuple, output_file_name: str = None) -> None
   :canonical: scan.init_pyfai_process_pool

   .. autodoc2-docstring:: scan.init_pyfai_process_pool

.. py:function:: pyfai_stat_exitangles(experiment, imageindex, scan, two_theta_start, pyfaiponi, anglimits, qmapbins, ivqbins) -> None
   :canonical: scan.pyfai_stat_exitangles

   .. autodoc2-docstring:: scan.pyfai_stat_exitangles

.. py:function:: pyfai_stat_qmap(experiment, imageindex, scan, two_theta_start, pyfaiponi, qlimits, qmapbins, ivqbins) -> None
   :canonical: scan.pyfai_stat_qmap

   .. autodoc2-docstring:: scan.pyfai_stat_qmap

.. py:function:: pyfai_stat_ivsq(experiment, imageindex, scan, two_theta_start, pyfaiponi, qmapbins, ivqbins) -> None
   :canonical: scan.pyfai_stat_ivsq

   .. autodoc2-docstring:: scan.pyfai_stat_ivsq

.. py:function:: pyfai_move_exitangles(experiment, imageindices, scan, shapecake, shapeqi, shapeexhexv, two_theta_start, pyfaiponi, anglimits, qmapbins) -> None
   :canonical: scan.pyfai_move_exitangles

   .. autodoc2-docstring:: scan.pyfai_move_exitangles

.. py:function:: pyfai_move_qmap(experiment, imageindices, scan, shapecake, shapeqi, shapeqpqp, two_theta_start, pyfaiponi, radrange, radstepval, qmapbins, qlimits=None) -> None
   :canonical: scan.pyfai_move_qmap

   .. autodoc2-docstring:: scan.pyfai_move_qmap

.. py:function:: pyfai_move_ivsq(experiment, imageindices, scan, shapecake, shapeqi, shapeqpqp, two_theta_start, pyfaiponi, radrange, radstepval, qmapbins) -> None
   :canonical: scan.pyfai_move_ivsq

   .. autodoc2-docstring:: scan.pyfai_move_ivsq

.. py:function:: _on_exit(shared_mem: multiprocessing.shared_memory.SharedMemory) -> None
   :canonical: scan._on_exit

   .. autodoc2-docstring:: scan._on_exit

.. py:function:: chunk(lst, num_chunks)
   :canonical: scan.chunk

   .. autodoc2-docstring:: scan.chunk

.. py:function:: _chunk_indices(array: numpy.ndarray, num_chunks: int) -> tuple
   :canonical: scan._chunk_indices

   .. autodoc2-docstring:: scan._chunk_indices

.. py:function:: _bin_one_map(start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, min_intensity: float, idx: int, processing_steps: list, oop: str, map_each_image: bool = False, previous_images: int = 0) -> numpy.ndarray
   :canonical: scan._bin_one_map

   .. autodoc2-docstring:: scan._bin_one_map

.. py:function:: bin_maps_with_indices(indices: typing.List[int], start: numpy.ndarray, stop: numpy.ndarray, step: numpy.ndarray, min_intensity: float, motors: typing.Dict[str, numpy.ndarray], metadata: dict, processing_steps: list, skip_images: typing.List[int], oop: str, map_each_image: bool = False, previous_images: int = 0) -> None
   :canonical: scan.bin_maps_with_indices

   .. autodoc2-docstring:: scan.bin_maps_with_indices

.. py:class:: Scan(metadata: fast_rsm.rsm_metadata.RSMMetadata, skip_images: typing.List[int] = None)
   :canonical: scan.Scan

   .. autodoc2-docstring:: scan.Scan

   .. rubric:: Initialization

   .. autodoc2-docstring:: scan.Scan.__init__

   .. py:property:: processing_steps
      :canonical: scan.Scan.processing_steps

      .. autodoc2-docstring:: scan.Scan.processing_steps

   .. py:method:: add_processing_step(function) -> None
      :canonical: scan.Scan.add_processing_step

      .. autodoc2-docstring:: scan.Scan.add_processing_step

   .. py:method:: load_image(idx: int, load_data=True)
      :canonical: scan.Scan.load_image

      .. autodoc2-docstring:: scan.Scan.load_image

   .. py:method:: q_bounds(frame: diffraction_utils.Frame, oop: str = 'y') -> typing.Tuple[numpy.ndarray]
      :canonical: scan.Scan.q_bounds

      .. autodoc2-docstring:: scan.Scan.q_bounds

   .. py:method:: from_i10(path_to_nx: typing.Union[str, pathlib.Path], beam_centre: typing.Tuple[int], detector_distance: float, sample_oop: diffraction_utils.Vector3, path_to_data: str = '')
      :canonical: scan.Scan.from_i10
      :classmethod:

      .. autodoc2-docstring:: scan.Scan.from_i10

   .. py:method:: from_i07(path_to_nx: typing.Union[str, pathlib.Path], beam_centre: typing.Tuple[int], detector_distance: float, setup: str, path_to_data: str = '')
      :canonical: scan.Scan.from_i07
      :staticmethod:

      .. autodoc2-docstring:: scan.Scan.from_i07
