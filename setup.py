"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import setuptools

setuptools.setup(
    name="kindle",
    version="0.1.3",
    packages=setuptools.find_packages(),
    description="PyTorch no-code model builder.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license=open("LICENSE").read().replace("\n", " "),
    author="Jongkuk Lim",
    author_email="lim.jeikei@gmail.com",
    url="https://github.com/JeiKeiLim/kindle",
    download_url="",
    install_requires=["tqdm>=4.56.0", "PyYAML>=5.3.1"],
    classifiers=["Programming Language :: Python :: 3.8"],
)
