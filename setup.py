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
    install_requires=[
        'matplotlib',
        'scikit-learn',
        'scipy',
        'tensorflow',
        'torch',
        'torchvision',
        'umap-learn',
        'unioncom',
    ],
    extras_require={
        'dev': [
            'flake8',
            'flake8-docstrings',
            'flake8-import-order',
            'pip-tools',
            'pytest',
            'pytest-cov',
        ],
        'notebooks': [
            'jupyterlab',
            'pandas',
        ],
    },
	tests_require=['pytest'],
)
