"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import setuptools

import kindle

setuptools.setup(
    name="kindle",
    version=kindle.__version__,
    packages=setuptools.find_packages(),
    description="PyTorch no-code model builder.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license=open("LICENSE").read().replace("\n", " "),
    author="Jongkuk Lim",
    author_email="lim.jeikei@gmail.com",
    url="https://github.com/JeiKeiLim/kindle",
    download_url="",
    install_requires=[
        "tqdm>=4.56.0",
        "PyYAML>=5.3.1",
        "torch>=1.7.1",
        "ptflops>=0.6.4",
        "timm>=0.4.5",
        "tabulate>=0.8.9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keyword="pytorch",
)
