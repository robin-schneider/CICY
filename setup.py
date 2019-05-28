import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCICY",
    version="0.01",
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
        "LICENSE :: OSI APPROVED :: GNU GENERAL PUBLIC LICENSE V3 (GPLV3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "texttable",
    ],
)
