"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import setuptools
from setuptools import setup

setup(
    name="torch-kindle",
    version="0.0.3",
    packages=setuptools.find_packages(),
    description="PyTorch no-code model builder.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license=open("LICENSE").read(),
    author="Jongkuk Lim",
    author_email="lim.jeikei@gmail.com",
    url="https://github.com/JeiKeiLim/torch-kindle",
    download_url="",
    install_requires=["tqdm>=4.56.0", "PyYAML>=5.3.1"],
    classifiers=["Programming Language :: Python :: 3.8"],
)
