from setuptools import find_packages, setup

with open('mmd_wrapper/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='mmd_wrapper',
    description=readme,
    version=__version__,
    packages=find_packages(exclude=('tests')),
    install_requires=[],
	tests_require=['pytest']
)
