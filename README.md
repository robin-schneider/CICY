# pyCICY - v0.02
A python CICY toolkit, which allows the computation of line bundle cohomologies over Complete Intersection Calabi Yau manifolds. It further contains functions for determining various topological quantities, such as Chern classes, triple intersection and Hodge numbers.

Installation is straighforwad with pip

```console
pip install pyCICY
```

or to always get the latest version
```console
pip install git+https://github.com/robin-schneider/CICY.git
```

# Changelog

v0.03 - Some bug fixes, more numpy

v0.02 - Major overhaul. Bug fixes, more numpy,
        some efficiency upgrade, improved logging,
		SpaSM support, relabeling of functions. 
		INFO: breaks backwards compatibility for some fcuntions.
		- 11.11.2019.

v0.01 - pyCICY toolkit made available - 29.5.2019.


# Quickstart
We import the CICY object from the module

```python
from pyCICY import CICY
```

Next we define a CICY, for example the tetraquadric:

```python
M = CICY([[1,2],[1,2],[1,2],[1,2]])
```

Now we are able to do some calculations, e.g.

```python
M.line_co([1,2,-4,1])
```

determines the line bundle cohomologies of the line bundle L = O(1,2,-4,1).

Since the rank computation takes the most time we included [SpasM - github](http://github.com/cbouilla/spasm). A working installation of SpaSM is required.

```python
T = CICY([[1,2,0,0,0],[1,0,2,0,0],[1,0,0,2,0],[1,0,0,0,2],[3,1,1,1,1]])
```

Next set the path to the SpaSM directory

```python
T.set_spasm_dir('/home/user/pathto/spasm/bench')
```

and do some computations:

```python
T.line_co([3,-4,2,3,5], SpaSM=True)
```


# Documentation

Documentation can be found on readthedocs [pyCICY](https://pycicy.readthedocs.io/en/latest/).


# Literature

The module has been developed in the context of the following paper:

```tex
@article{Larfors:2019sie,
      author         = "Larfors, Magdalena and Schneider, Robin",
      title          = "{Line bundle cohomologies on CICYs with Picard number
                        two}",
      year           = "2019",
      eprint         = "1906.00392",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-th",
      reportNumber   = "UUITP-18/19",
      SLACcitation   = "%%CITATION = ARXIV:1906.00392;%%"
}
````

Further literature can be found here:

```tex
@book{Hubsch:1992nu,
	author         = "Hubsch, Tristan",
	title          = "{Calabi-Yau manifolds: A Bestiary for physicists}",
	publisher      = "World Scientific",
	address        = "Singapore",
	year           = "1994",
	ISBN           = "9789810219277, 981021927X",
	SLACcitation   = "%%CITATION = INSPIRE-338506;%%"
}

@phdthesis{Anderson:2008ex,
	author         = "Anderson, Lara Briana",
	title          = "{Heterotic and M-theory Compactifications for String
	Phenomenology}",
	school         = "Oxford U.",
	url            = "https://inspirehep.net/record/793857/files/arXiv:0808.3621.pdf",
	year           = "2008",
	eprint         = "0808.3621",
	archivePrefix  = "arXiv",
	primaryClass   = "hep-th",
	SLACcitation   = "%%CITATION = ARXIV:0808.3621;%%"
}
```

The SpaSM library can be found here: [github](http://github.com/cbouilla/spasm)
```tex
@manual{spasm,
title = {{SpaSM}: a Sparse direct Solver Modulo $p$},
author = {The SpaSM group},
edition = {v1.2},
year = {2017},
note = {\url{http://github.com/cbouilla/spasm}}
}
```

# Useful software

The pyCICY module is (mostly) backwards compatible with python 2.7. Hence, it can be used in [Sage](http://www.sagemath.org/). Other useful packages for dealing with Calabi Yau manifolds in toric varieties are [cohomCalg](https://github.com/BenjaminJurke/cohomCalg/) and [PALP](http://hep.itp.tuwien.ac.at/~kreuzer/CY/CYpalp.html).
