import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCICY",
    version="0.5.1",
    author="Robin Schneider",
    author_email="robin.schneider@physics.uu.se",
    description="A python CICY toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robin-schneider/CICY",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "texttable",
        "matplotlib",
    ],
)
