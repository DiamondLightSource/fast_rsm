:py:mod:`experiment`
====================

.. py:module:: experiment

.. autodoc2-docstring:: experiment
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Experiment <experiment.Experiment>`
     - .. autodoc2-docstring:: experiment.Experiment
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_remove_file <experiment._remove_file>`
     - .. autodoc2-docstring:: experiment._remove_file
          :summary:
   * - :py:obj:`_sum_numpy_files <experiment._sum_numpy_files>`
     - .. autodoc2-docstring:: experiment._sum_numpy_files
          :summary:
   * - :py:obj:`_q_to_theta <experiment._q_to_theta>`
     - .. autodoc2-docstring:: experiment._q_to_theta
          :summary:
   * - :py:obj:`_match_start_stop_to_step <experiment._match_start_stop_to_step>`
     - .. autodoc2-docstring:: experiment._match_start_stop_to_step
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`combine_ranges <experiment.combine_ranges>`
     - .. autodoc2-docstring:: experiment.combine_ranges
          :summary:

API
~~~

.. py:data:: combine_ranges
   :canonical: experiment.combine_ranges
   :value: None

   .. autodoc2-docstring:: experiment.combine_ranges

.. py:function:: _remove_file(path: typing.Union[str, pathlib.Path])
   :canonical: experiment._remove_file

   .. autodoc2-docstring:: experiment._remove_file

.. py:function:: _sum_numpy_files(filenames: typing.List[typing.Union[pathlib.Path, str]])
   :canonical: experiment._sum_numpy_files

   .. autodoc2-docstring:: experiment._sum_numpy_files

.. py:function:: _q_to_theta(q_values, energy) -> numpy.array
   :canonical: experiment._q_to_theta

   .. autodoc2-docstring:: experiment._q_to_theta

.. py:class:: Experiment(scans: typing.List[fast_rsm.scan.Scan])
   :canonical: experiment.Experiment

   .. autodoc2-docstring:: experiment.Experiment

   .. rubric:: Initialization

   .. autodoc2-docstring:: experiment.Experiment.__init__

   .. py:method:: _clean_temp_files() -> None
      :canonical: experiment.Experiment._clean_temp_files

      .. autodoc2-docstring:: experiment.Experiment._clean_temp_files

   .. py:method:: add_processing_step(processing_step: callable) -> None
      :canonical: experiment.Experiment.add_processing_step

      .. autodoc2-docstring:: experiment.Experiment.add_processing_step

   .. py:method:: mask_pixels(pixels: tuple) -> None
      :canonical: experiment.Experiment.mask_pixels

      .. autodoc2-docstring:: experiment.Experiment.mask_pixels

   .. py:method:: mask_regions(regions: typing.List[diffraction_utils.Region])
      :canonical: experiment.Experiment.mask_regions

      .. autodoc2-docstring:: experiment.Experiment.mask_regions

   .. py:method:: mask_edf(edfmask)
      :canonical: experiment.Experiment.mask_edf

      .. autodoc2-docstring:: experiment.Experiment.mask_edf

   .. py:method:: binned_reciprocal_space_map(num_threads: int, map_frame: diffraction_utils.Frame, output_file_name: str = 'mapped', min_intensity_mask: float = None, output_file_size: float = 100, save_vtk: bool = True, save_npy: bool = True, oop: str = 'y', volume_start: numpy.ndarray = None, volume_stop: numpy.ndarray = None, volume_step: numpy.ndarray = None, map_each_image: bool = False)
      :canonical: experiment.Experiment.binned_reciprocal_space_map

      .. autodoc2-docstring:: experiment.Experiment.binned_reciprocal_space_map

   .. py:method:: _project_to_1d(num_threads: int, output_file_name: str = 'mapped', num_bins: int = 1000, bin_size: float = None, oop: str = 'y', tth=False, only_l=False)
      :canonical: experiment.Experiment._project_to_1d

      .. autodoc2-docstring:: experiment.Experiment._project_to_1d

   .. py:method:: intensity_vs_l(num_threads: int, output_file_name: str = 'mapped', num_bins: int = 1000, bin_size: float = None, oop: str = 'y')
      :canonical: experiment.Experiment.intensity_vs_l

      .. autodoc2-docstring:: experiment.Experiment.intensity_vs_l

   .. py:method:: intensity_vs_tth(num_threads: int, output_file_name: str = 'mapped', num_bins: int = 1000, bin_size: float = None, oop: str = 'y')
      :canonical: experiment.Experiment.intensity_vs_tth

      .. autodoc2-docstring:: experiment.Experiment.intensity_vs_tth

   .. py:method:: intensity_vs_q(num_threads: int, output_file_name: str = 'I vs Q', num_bins: int = 1000, bin_size: float = None, oop: str = 'y')
      :canonical: experiment.Experiment.intensity_vs_q

      .. autodoc2-docstring:: experiment.Experiment.intensity_vs_q

   .. py:method:: q_bounds(frame: diffraction_utils.Frame, oop: str = 'y') -> typing.Tuple[numpy.ndarray]
      :canonical: experiment.Experiment.q_bounds

      .. autodoc2-docstring:: experiment.Experiment.q_bounds

   .. py:method:: calcq(twotheta, wavelength)
      :canonical: experiment.Experiment.calcq

      .. autodoc2-docstring:: experiment.Experiment.calcq

   .. py:method:: calcqstep(gammastep, gammastart, wavelength)
      :canonical: experiment.Experiment.calcqstep

      .. autodoc2-docstring:: experiment.Experiment.calcqstep

   .. py:method:: histogram_xy(x, y, step_size)
      :canonical: experiment.Experiment.histogram_xy

      .. autodoc2-docstring:: experiment.Experiment.histogram_xy

   .. py:method:: SOHqcalc(angle, kmod)
      :canonical: experiment.Experiment.SOHqcalc

      .. autodoc2-docstring:: experiment.Experiment.SOHqcalc

   .. py:method:: calcanglim(axis, vertsetup=False)
      :canonical: experiment.Experiment.calcanglim

      .. autodoc2-docstring:: experiment.Experiment.calcanglim

   .. py:method:: calcqlim(axis, vertsetup=False)
      :canonical: experiment.Experiment.calcqlim

      .. autodoc2-docstring:: experiment.Experiment.calcqlim

   .. py:method:: do_savetiffs(hf, data, axespara, axesperp)
      :canonical: experiment.Experiment.do_savetiffs

      .. autodoc2-docstring:: experiment.Experiment.do_savetiffs

   .. py:method:: do_savedats(hf, Idata, qdata, tthdata)
      :canonical: experiment.Experiment.do_savedats

      .. autodoc2-docstring:: experiment.Experiment.do_savedats

   .. py:method:: pyfai_stat_exitangles_wrapper(args)
      :canonical: experiment.Experiment.pyfai_stat_exitangles_wrapper

      .. autodoc2-docstring:: experiment.Experiment.pyfai_stat_exitangles_wrapper

   .. py:method:: pyfai_stat_qmap_wrapper(args)
      :canonical: experiment.Experiment.pyfai_stat_qmap_wrapper

      .. autodoc2-docstring:: experiment.Experiment.pyfai_stat_qmap_wrapper

   .. py:method:: pyfai_stat_ivsq_wrapper(args)
      :canonical: experiment.Experiment.pyfai_stat_ivsq_wrapper

      .. autodoc2-docstring:: experiment.Experiment.pyfai_stat_ivsq_wrapper

   .. py:method:: pyfai_static_exitangles(hf, scan, num_threads, pyfaiponi, ivqbins, qmapbins=0)
      :canonical: experiment.Experiment.pyfai_static_exitangles

      .. autodoc2-docstring:: experiment.Experiment.pyfai_static_exitangles

   .. py:method:: pyfai_static_qmap(hf, scan, num_threads, output_file_path, pyfaiponi, ivqbins, qmapbins=0)
      :canonical: experiment.Experiment.pyfai_static_qmap

      .. autodoc2-docstring:: experiment.Experiment.pyfai_static_qmap

   .. py:method:: get_bin_axvals(data_in, ind)
      :canonical: experiment.Experiment.get_bin_axvals

      .. autodoc2-docstring:: experiment.Experiment.get_bin_axvals

   .. py:method:: pyfai_static_ivsq(hf, scan, num_threads, output_file_path, pyfaiponi, ivqbins, qmapbins=0)
      :canonical: experiment.Experiment.pyfai_static_ivsq

      .. autodoc2-docstring:: experiment.Experiment.pyfai_static_ivsq

   .. py:method:: calcnewrange(range2)
      :canonical: experiment.Experiment.calcnewrange

      .. autodoc2-docstring:: experiment.Experiment.calcnewrange

   .. py:method:: pyfai_moving_ivsq(hf, scanlist, num_threads, output_file_path, pyfaiponi, radrange, radstepval, qmapbins=0)
      :canonical: experiment.Experiment.pyfai_moving_ivsq

      .. autodoc2-docstring:: experiment.Experiment.pyfai_moving_ivsq

   .. py:method:: pyfai_moving_exitangles(hf, scanlist, num_threads, output_file_path, pyfaiponi, radrange, radstepval, qmapbins=0)
      :canonical: experiment.Experiment.pyfai_moving_exitangles

      .. autodoc2-docstring:: experiment.Experiment.pyfai_moving_exitangles

   .. py:method:: pyfai_moving_qmap(hf, scanlist, num_threads, output_file_path, pyfaiponi, radrange, radstepval, qmapbins=0)
      :canonical: experiment.Experiment.pyfai_moving_qmap

      .. autodoc2-docstring:: experiment.Experiment.pyfai_moving_qmap

   .. py:method:: gamdel2rots(gamma, delta)
      :canonical: experiment.Experiment.gamdel2rots

      .. autodoc2-docstring:: experiment.Experiment.gamdel2rots

   .. py:method:: load_curve_values(scan)
      :canonical: experiment.Experiment.load_curve_values

      .. autodoc2-docstring:: experiment.Experiment.load_curve_values

   .. py:method:: createponi(outpath, image2dshape, beam_centre=0, offset=0)
      :canonical: experiment.Experiment.createponi

      .. autodoc2-docstring:: experiment.Experiment.createponi

   .. py:method:: save_projection(hf, projected2d, twothetas, Qangs, intensities, config)
      :canonical: experiment.Experiment.save_projection

      .. autodoc2-docstring:: experiment.Experiment.save_projection

   .. py:method:: save_integration(hf, twothetas, Qangs, intensities, configs, scan=0)
      :canonical: experiment.Experiment.save_integration

      .. autodoc2-docstring:: experiment.Experiment.save_integration

   .. py:method:: save_qperp_qpara(hf, qperp_qpara_map, scan=0)
      :canonical: experiment.Experiment.save_qperp_qpara

      .. autodoc2-docstring:: experiment.Experiment.save_qperp_qpara

   .. py:method:: save_config_variables(hf, joblines, pythonlocation, globalvals)
      :canonical: experiment.Experiment.save_config_variables

      .. autodoc2-docstring:: experiment.Experiment.save_config_variables

   .. py:method:: reshape_to_signalshape(arr, signal_shape)
      :canonical: experiment.Experiment.reshape_to_signalshape

      .. autodoc2-docstring:: experiment.Experiment.reshape_to_signalshape

   .. py:method:: save_scan_field_values(hf, scan)
      :canonical: experiment.Experiment.save_scan_field_values

      .. autodoc2-docstring:: experiment.Experiment.save_scan_field_values

   .. py:method:: deprecation_msg(option)
      :canonical: experiment.Experiment.deprecation_msg

      .. autodoc2-docstring:: experiment.Experiment.deprecation_msg

   .. py:method:: from_i07_nxs(nexus_paths: typing.List[typing.Union[str, pathlib.Path]], beam_centre: typing.Tuple[int], detector_distance: float, setup: str, path_to_data: str = '', using_dps: bool = False, experimental_hutch=0)
      :canonical: experiment.Experiment.from_i07_nxs
      :classmethod:

      .. autodoc2-docstring:: experiment.Experiment.from_i07_nxs

.. py:function:: _match_start_stop_to_step(step, user_bounds, auto_bounds, eps=1e-05)
   :canonical: experiment._match_start_stop_to_step

   .. autodoc2-docstring:: experiment._match_start_stop_to_step
