:py:mod:`meta_analysis`
=======================

.. py:module:: meta_analysis

.. autodoc2-docstring:: meta_analysis
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_step_from_filesize <meta_analysis.get_step_from_filesize>`
     - .. autodoc2-docstring:: meta_analysis.get_step_from_filesize
          :summary:
   * - :py:obj:`_find_exc_broken_frames <meta_analysis._find_exc_broken_frames>`
     - .. autodoc2-docstring:: meta_analysis._find_exc_broken_frames
          :summary:
   * - :py:obj:`skip_i07_exc_broken_frames <meta_analysis.skip_i07_exc_broken_frames>`
     - .. autodoc2-docstring:: meta_analysis.skip_i07_exc_broken_frames
          :summary:

API
~~~

.. py:function:: get_step_from_filesize(start: numpy.ndarray, stop: numpy.ndarray, file_size: float = 100) -> numpy.ndarray
   :canonical: meta_analysis.get_step_from_filesize

   .. autodoc2-docstring:: meta_analysis.get_step_from_filesize

.. py:function:: _find_exc_broken_frames(scan: fast_rsm.scan.Scan)
   :canonical: meta_analysis._find_exc_broken_frames

   .. autodoc2-docstring:: meta_analysis._find_exc_broken_frames

.. py:function:: skip_i07_exc_broken_frames(experiment: fast_rsm.experiment.Experiment)
   :canonical: meta_analysis.skip_i07_exc_broken_frames

   .. autodoc2-docstring:: meta_analysis.skip_i07_exc_broken_frames
