from setuptools import setup, find_packages

requires = [
    'holidays',
    'numpy',
    'openrouteservice',
    'pandas',
]

# package_data = {'forest.poplar.raw': ['noncode/*.csv', 'noncode/*.json']}

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='MobileDataToolkit',
    version='0.0.0',
    description='--add description here--',
    long_description=readme,
    author='Ekin Ugurel',
    author_email='ugurel@uw.com',
    license=license,
    #packages=find_packages(include=["forest*"]),
    #package_data=package_data,
    install_requires=requires
)