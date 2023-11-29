"""The QSweep Synthesis Setup File."""
from __future__ import annotations

import os

from setuptools import find_packages
from setuptools import setup

root_dir_path = os.path.abspath(os.path.dirname(__file__))
pkg_dir_path = os.path.join(root_dir_path, 'qsweep')
readme_path = os.path.join(root_dir_path, 'README.md')
version_path = os.path.join(pkg_dir_path, 'version.py')

# Load Version Number
with open(version_path) as version_file:
    exec(version_file.read())

# Load Readme
with open(readme_path) as readme_file:
    long_description = readme_file.read()

setup(
    name='qsweep',
    version=__version__,  # type: ignore # noqa # Defined in version.py loaded above
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
    packages=find_packages(exclude=['examples*', 'test*']),
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
