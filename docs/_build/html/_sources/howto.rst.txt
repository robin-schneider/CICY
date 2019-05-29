How to use pyCICY
-----------------

Using pyCICY is straightforward. We import the CICY object

>>> from pyCICY import CICY

and can then directly start by defining some manifolds.
For example the famous quintic:

>>> Q = CICY('quintic', [[4,5]])

or the tetra quadric:

>>> T = CICY('tetra', [[1,2],[1,2],[1,2],[1,2]])

or the manifold #7833 of the CICYlist:

>>> M = CICY('7833', [[2,2,1],[3,1,3]])

Calculating line bundle cohomologies is done via

>>> M.line_co([-3,4])
[0, 87, 0, 0]

the toolkit also includes various other functions. An incomplete overview is given by

>>> M.help()
...

for a full list, consult the documentation.
