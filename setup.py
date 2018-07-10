from pathlib import Path

from importlib.machinery import SourceFileLoader
from setuptools import setup

readme = Path(__file__).parent.joinpath('README.md')
if readme.exists():
    with readme.open() as f:
        long_description = f.read()
else:
    long_description = '-'

setup(
    name='serving-utils',
    version='0.1.0',
    description='Some utilities for tensorflow serving',
    long_description=long_description,
    python_requires='>=3.6',
    packages=[
        'serving-utils',
    ],
    author='Po-Hsien Chu',
    author_email='cph@yoctol.com',
    url='https://github.com/samuelcolvin/arq',
    license='MIT',
    install_requires=[],
)
