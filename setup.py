"""
The setup file. To install as a developer (only do this in a venv):

python setup.py develop

Otherwise, just:

python setup.py install
"""

from setuptools import setup, find_packages

setup(
    name='fast_rsm',
    version='1.2.1',
    license='MIT License',
    packages=find_packages('src'),
    description=(
        "A python package for analysing surface x-ray diffraction images " +
        "and map into reciprocal space volumes."),
    author='Philip Mousley',
    author_email='philip.mousley@diamond.ac.uk',
    package_dir={'': 'src'},
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    # The bare minimum for installation.
    install_requires=["scipy >= 1.8.0"]
)
