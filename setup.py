#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'calo-ddpm',
    version          = '0.0.1',
    author           = 'The LS4GAN Project Developers',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "Full Detector Simulation of Heavy Ion Collisions",
    packages         = setuptools.find_packages(
        include = [ 'jetgen', 'jetgen.*' ]
    ),
    install_requires = [ 'numpy', 'pandas', 'tqdm', 'Pillow' ],
)

