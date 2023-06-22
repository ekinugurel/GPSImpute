from setuptools import setup, find_packages

requires = [
    'holidays',
    'numpy',
    'openrouteservice',
    'pandas',
    'gpytorch',
    'matplotlib',
    'scikit-mobility',
    'torchaudio',
    'torchvision'
]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='GPSImpute',
    version='0.0.0',
    description='--add description here--',
    long_description=readme,
    author='Ekin Ugurel',
    author_email='ugurel@uw.com',
    license=license,
    packages=find_packages(),
    #package_data=package_data,
    install_requires=requires
)
