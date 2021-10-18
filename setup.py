"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import setuptools

import kindle

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("LICENSE", encoding="utf-8") as f:
    license_content = f.read().replace("\n", " ")

setuptools.setup(
    name="kindle",
    version=kindle.__version__,
    packages=setuptools.find_packages(),
    description="Kindle - Making a PyTorch model easier than ever!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=license_content,
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
        "einops>=0.3.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keyword="pytorch",
)
