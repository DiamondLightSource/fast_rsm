[![Total alerts](https://img.shields.io/lgtm/alerts/g/RBrearton/RSMapper.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RBrearton/RSMapper/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/RBrearton/RSMapper.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RBrearton/RSMapper/context:python)
[![codecov](https://codecov.io/gh/RBrearton/RSMapper/branch/master/graph/badge.svg?token=FGIV0MVHS8)](https://codecov.io/gh/RBrearton/RSMapper)
[![Actions Status](https://github.com/RBrearton/RSMapper/workflows/pytest/badge.svg)](https://github.com/pytest/RSMapper/actions)
[![Documentation Status](https://readthedocs.org/projects/local-stats/badge/?version=latest)](https://local-stats.readthedocs.io/en/latest/?badge=latest)

A fast reciprocal space mapper.

This module depends on two of my other modules:
https://github.com/RBrearton/diffraction_utils
is used for the parsing of awkward data formats and for the generation of many
of the scipy.spatial.transform.Rotation objects that are used in the
calculations. The idea is that https://github.com/RBrearton/diffraction_utils is
a general purpose tool that can be used in many contexts besides purely
reciprocal space mapping, although it originated as a spin-out of this project.

https://github.com/RBrearton/Custom-bin contains the binning algorithm used by
default, although several more are included in the RSMapper.binning module.
