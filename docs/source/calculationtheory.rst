Theory for mapping calculations
=================================


Full reciprocal space maps
--------------------------------

The mapping process for full reciprocal maps is essentially made up of three steps:

- work out the exit vector to each pixel on the detector
- from these exit vectors calculate the HKL vectors to each pixel
- use these HKL vectors to bin the pixel intensities into the specified HKL resolution



Calculation of exit vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This uses the values of delta and gamma to rotate the exitvector to the detectors position, and then uses the beam centre to positions the image correctly. 

