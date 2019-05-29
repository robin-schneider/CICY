# pyCICY
A python CICY toolkit, which allows the computation of line bundle cohomologies over Complete Intersection Calabi Yau manifolds. It further contains functions for determining various topological quantities, such as Chern classes, triple intersection and Hodge numbers.

Installation is straighforwad with pip

```console
pip install pyCICY
```

The module has been developed in the context of the following paper:

```tex
@article{Larfors:2019xxx,
	author         = "Larfors, Magdalena and Schneider, Robin",
	title          = "{Line bundle cohomologies on CICYs with Picard number two}",
	year           = "2019",
	eprint         = "1905.XXXXX",
	archivePrefix  = "arXiv",
	primaryClass   = "hep-th",
	SLACcitation   = "%%CITATION = ARXIV:1905.XXXXX;%%"
}
````

It is based on methods developed in

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
````

# Quickstart
We import the CICY object from the module

```python
from pyCICY import CICY
```

Next we define a CICY, for example the tetraquadric:

```python
M = CICY('tetra', [[1,2],[1,2],[1,2],[1,2]])
```

Now we are able to do some calculations, e.g.

```python
M.line_co([1,2,-4,1])
```

determines the line bundle cohomologies of the line bundle [$]\mathcal{O}(1,2,-4,1)[\$].


# Changelog

v0.01 - pyCICY toolkit made available - 29.5.2019.

# Useful software

pyCICY is backwards compatible with python 2.7. Hence, it can be used in [Sage](http://www.sagemath.org/). Other useful packages for dealing with Calabi Yau manifolds in toric varieties are [cohomCalg](https://github.com/BenjaminJurke/cohomCalg/) and [PALP](http://hep.itp.tuwien.ac.at/~kreuzer/CY/CYpalp.html).
