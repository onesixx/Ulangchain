# Licensed under the MIT License.
# Copyright (c) Sixx.

# from rosie.log import setup_logging
# from rosie.log import logger
# logger.info("Here we go!!!")

from setuptools import setup, find_packages

# with open("requirements.txt") as f:
#     required_packages = f.read().splitlines()
setup(
    name='rosie',
    version='0.1',
    author='onesixx',
    author_email='onesixx@nate.com',
    description='A Python package for AI',
    # long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/onesixx',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
    #     'numpy>=1.24.3',
    #     'pandas>=2.0.1',
    #     'matplotlib>=3.7.1',
    ],
)
