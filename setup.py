from pathlib import Path

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

readme = Path(__file__).parent.joinpath('README.md')
if readme.exists():
    with readme.open() as f:
        long_description = f.read()
else:
    long_description = '-'


class BuildPackageProtos(install):
    def run(self):
        install.run(self)
        from grpc.tools import command
        command.build_package_protos('')


class BuildPackageProtosDevelop(develop):
    def run(self):
        develop.run(self)
        from grpc.tools import command
        command.build_package_protos('')


setup(
    name='serving-utils',
    version='0.1.0',
    description='Some utilities for tensorflow serving',
    long_description=long_description,
    python_requires='>=3.5',
    packages=[
        'serving_utils',
    ],
    author='Po-Hsien Chu',
    author_email='cph@yoctol.com',
    url='https://github.com/samuelcolvin/arq',
    license='MIT',
    setup_requires=[
        'cython',
    ],
    install_requires=[
        'grpcio-tools',
        'aiogrpc>=1.5',
        'numpy>=1.14.0',
    ],
    cmdclass={
        'install': BuildPackageProtos,
        'develop': BuildPackageProtosDevelop,
    },
)
