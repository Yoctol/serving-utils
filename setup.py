from pathlib import Path

from setuptools import setup

readme = Path(__file__).parent.joinpath('README.md')
if readme.exists():
    with readme.open() as f:
        long_description = f.read()
        try:
            from pypandoc import convert_text
            long_description = convert_text(long_description, 'rst', format='md')
        except ImportError:
            print("warning: pypandoc module not found, could not convert Markdown to RST")
else:
    long_description = '-'

setup(
    name='serving-utils',
    version='0.9.1',
    description='Some utilities for tensorflow serving',
    long_description=long_description,
    python_requires='>=3.6',
    packages=[
        'serving_utils',
        'serving_utils.protos',
    ],
    author='Po-Hsien Chu',
    author_email='cph@yoctol.com',
    url='https://github.com/Yoctol/serving-utils',
    license='MIT',
    setup_requires=[
        'cython',
        'pypandoc',
    ],
    install_requires=[
        'grpclib',
        'grpcio-tools',
        'numpy>=1.14.0',
    ],
)
