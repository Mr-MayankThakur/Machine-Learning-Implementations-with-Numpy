import os
from io import open

from setuptools import setup, find_packages

ROOT_FOLDER = os.path.abspath(os.path.dirname(__name__))

# Specify which minimum python version this project supports.
# Make sure you update the classifiers list later in this file to match the version requirement here.
PYTHON_VERSION = '>=3.6'

# The project dependencies are listed in an anaconda environment file. That file is only used for development.
# To run the project on any computer other than your local machine or the build server, please include the dependencies
DEPENDENCIES = [
    'click>=7.0',
    'numpy>=1.16',
    'scipy>=1.1',
    'scikit-learn>=0.19',
    'requests>=2.20'
]

# The version for the project is specified in the machine_learning_implementations_with_numpy/__version__.py
# The next bit of code reads this file and converts it into a constant.
about = {}

with open(os.path.join(ROOT_FOLDER, 'machine_learning_implementations_with_numpy', '__version__.py')) as version_file:
    VERSION = exec(version_file.read(), about)

# The long description for the project comes from its README.rst file in the root directory.
# Please make sure that the README.rst file is up-to-date before publishing the project.
with open(os.path.join(ROOT_FOLDER, 'README.rst')) as readme_file:
    long_description = readme_file.read()

setup(
    name='machine_learning_implementations_with_numpy',
    version=about['__version__'],
    description='These are the implementations of machine learning models using numpy and other basic modules.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='http://github.com/Mr-MayankThakur/machine_learning_implementations_with_numpy',

    author='Mr-MayankThakur',
    author_email='mackv210w@gmail.com',

    # Specify the classifiers for the project. This makes it easier to find the project once you publish on pipy.
    # This is optional, but highly recommended.
    #
    # For a list of valid classifier please see https://pypi.org/classifiers/
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate the target audience of the project
        'Intended Audience :: Developers',

        # Please specify the licens that you wnat to use.for
        'License :: OSI Approved :: Apache Software License',

        # Please include the python versions you're actually using.
        # For reference, we've added all versions of python 3 currently in use.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # Although the license is included in the classifiers you're required to set it again in the license field.
    license='Apache-2.0',

    # Please specify the keywords to find the project (optional).
    keywords='machine-learning data-science',

    # This next bit specifies the technical metadata for the package.
    # It includes the packages, required packages for installation and the supported python version.
    packages=find_packages(exclude=['docs', 'features', 'notebooks', 'outputs', 'tests']),
    install_requires=DEPENDENCIES,
    python_requires=PYTHON_VERSION,
    include_package_data=True,

    # We like to use Click as a tool to generate a CLI for the package.
    # You can disable this if you want, but it is recommended to leave it in.
    entry_points={
        'console_scripts': [
            'machine_learning_implementations_with_numpy=machine_learning_implementations_with_numpy.commandline:cli'
        ]
    }
)
