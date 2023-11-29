"""The QSweep Synthesis Setup File."""
from __future__ import annotations

import os

from setuptools import setup

root_dir_path = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(root_dir_path, 'README.md')

# Load Readme
with open(readme_path) as readme_file:
    long_description = readme_file.read()

setup(
    name='qsweep',
    version='0.1.0',
    description='QSweep Single-Qudit Synthesis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/edyounis/QSweep',
    author='Ed Younis',
    author_email='edyounis@lbl.gov',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Compilers',
        'Typing :: Typed',
    ],
    keywords=[
        'bqskit',
        'quantum',
        'synthesis',
        'single-qudit',
    ],
    project_urls={
        'Bug Tracker': 'https://github.com/edyounis/QSweep/issues',
        'Source Code': 'https://github.com/edyounis/QSweep',
    },
    py_modules=['qsweep'],
    install_requires=[
        'bqskit>=1.1.1',
    ],
    python_requires='>=3.8, <4',
    extras_require={
        'dev': [
            'pre-commit',
            'pytest',
        ],
    },
)
