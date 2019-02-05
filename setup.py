from setuptools import setup, find_packages

setup(
    name='PracticalDeepStereo',
    version='0.1',
    packages=find_packages(where="src", exclude=("test", )),
    package_dir={"": "./"},
)