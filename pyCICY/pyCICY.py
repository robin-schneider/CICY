"""
Created on Fri Jun 08 16:45:23 2018

pyCICY - A python CICY toolkit. It allows for computation of 
line bundle cohomologies over Complete Intersection
Calabi Yau manifolds.

Further, it includes functions to determine various 
topological quantities, such as Chern classes, Hodge numbers
and triple intersection numbers.

Authors
-------
Magdalena Larfors (magdalena.larfors@physics.uu.se)
Robin Schneider (robin.schneider@physics.uu.se)

Version
-------
v0.02 - Major overhaul. Bug fixes, more numpy,
        some efficiency upgrade, additional functions - 4.11.2019.
v0.01 - pyCICY toolkit made available - 29.5.2019.

"""

# libraries
import itertools
import numpy as np
import sympy as sp
import random
from random import randint
import scipy as sc
import scipy.special
import math
import time
from texttable import Texttable
import os
# for documentation
import logging
# for SpaSM
import subprocess
import tempfile
#cython map creation

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('pyCICY')

class CICY:

    def __init__(self, conf, log=3):
        """
        The CICY class. It allows for computation of various topological quantities.
        Main function is the computation of line bundle cohomologies.
        
        Parameters
        ----------
        conf : array[nProj, K+1]
            The CICY configuration matrix, including the first col for the
            projective spaces.
        log : int, optional
            Documentation level. 1 - DEBUG, 2 - INFO, 3 - WARNING, by default 3.

        Examples
        --------
        The famous quintic:

        >>> Q = CICY([[4,5]])

        or the tetra quadric:

        >>> T = CICY([[1,2],[1,2],[1,2],[1,2]])

        or the manifold #7833 of the CICYlist:

        >>> M = CICY([[2,2,1],[3,1,3]])
        """
        start = time.time()
        self.debug = False

        if log == 1:
            level = logging.DEBUG
        elif log == 2:
            level = logging.INFO
        elif log == 3:
            level = logging.WARNING

        logger.setLevel(level=level)

        # controls that no logger is done for init
        self.doc = False
        
        # define some variables which will make our life easier
        self.M = np.array(conf, dtype=np.int16)
        self.len = len(conf) # = #projective spaces
        self.K = len(conf[0])-1 # = #hyper surfaces
        self.dimA = sum([self.M[i][0] for i in range(self.len)])
        self.nfold = self.dimA-self.K
        self.N = np.array([[self.M[i][j] for j in range(1, self.K+1)] for i in range(self.len)])

        # check if actually CICY
        if not np.array_equal(self.first_chern(), np.zeros(self.len)):
            self.CY = False
            logger.warning('The configuration matrix does not belong to a Calabi Yau. '+
                    'Its first Chern class is {}.'.format(self.first_chern()))
            logger.warning('There is no official support for more general hypersurfaces.')
        else:
            self.CY = True

        #some topological quantities, which we only want to calculate once
        self.euler = 1 #set to 1 since all CICYs have negative.
        if self.nfold == 3:
            self.triple = np.array([])
            self.triple = self.triple_intersection()
        if self.nfold == 4:
            self.quadruple = np.array([])
            self.quadruple = self.quadruple_intersection()
        # need to define before hodge
        self.fav = False
        # defining normal bundle sections
        self.pmoduli, self.moduli = self._fill_moduli(9) 
        
        # needed for quick calculations of index
        self.c2_tensor = self.second_chern_all()

        self.h = self.hodge_data()

        # check if favourable, since needed for vanishing
        # theorem in line_co.
        if self.nfold == 3:
            if self.h[2] == self.len:
                self.fav = True
            else:
                self.fav = False
        elif self.nfold == 4:
            if self.h[3] == self.len:
                self.fav = True
            else:
                self.fav = False

        end = time.time()
        logger.info('Initialization took: {}.'.format(end-start))
        self.doc = True

        # artifacts from debugging; 
        # might be useful if you want to change some code
        self.name = '_name'
        self.directory = os.path.join(os.path.expanduser('~/Documents/data/CICY/'), self.name)
        self.debug = False

    def info(self):
        """
        Prints a broad overview of the geometric properties into the console.
        Includes: Configuration matrix, Hodge diamond, triple intersection
        numbers, Chern classes, Euler characteristic and defining
        Polynomials.
        """
        if self.nfold == 2:
            print('K3.')
        elif self.nfold == 3:
            print('The CICY {}'.format(self.M))
            print('has Hodge numbers {}.'.format(self.hodge_numbers()))
            print('Its triple intersetion numbers are \n {}.'.format(self.triple_intersection()))
            print('The second Chern class is {}.'.format(self.second_chern()))
            print('The euler characteristic is {}'.format(self.euler_characteristic()))
            print('Finally, its defining polynomials have been choosen to be \n {}.'.format(self.def_poly()))
        elif self.nfold == 4:
            print('The CICY {}'.format(self.M))
            print('has Hodge numbers {}.'.format(self.hodge_numbers()))
            print('Itsquadruple intersetion numbers are \n {}.'.format(self.quadruple_intersection()))
            print('The second Chern class is {}.'.format(self.second_chern()))
            print('The euler characteristic is {}'.format(self.euler_characteristic()))
            print('Finally, its defining polynomials have been choosen to be \n {}.'.format(self.def_poly()))

    def hodge_numbers(self):
        """
        Prints the hodge numbers into the console, only supported for
        2,3,4 folds.
        """
        if self.nfold == 2:
            print('h^{1,1} = 20')
        if self.nfold == 3:
            print('h^{1,1} = '+str(self.h[2]))
            print('h^{2,1} = '+str(self.h[1]))
        if self.nfold == 4:
            print('h^{1,1} = '+str(self.h[3]))
            print('h^{2,1} = '+str(self.h[2]))
            print('h^{2,2} = '+str(self.h[4]))
            print('h^{3,1} = '+str(self.h[1]))

    def def_poly(self):
        r"""
        Returns the defining polynomials with (redundant) complex moduli as coefficients.
        
        Returns
        -------
        normal: list/sympyexpr
            A list of the normal sections in terms of monomials.

        Example
        -------
        >>> M = CICY([[4,5]])
        >>> M.def_poly()
        [45*x0**5 + 50*x0**4*x1 + ... + 40*x3*x4**4 + 30*x4**5]
        """
        normal = [0 for _ in range(self.K)]
        projx = sp.symbols('x0:'+str(self.dimA+self.len), integer=True)
        # run over all normal bundle sections
        for i in range(self.K):
            # run over all monomials
            for j in range(len(self.moduli[i])):
                # define monomials
                tmp = self.moduli[i][j]
                for k in range(self.dimA+self.len):
                    tmp *= projx[k]**self.pmoduli[i][j][k]
                # add it to the respective entry
                normal[i] += tmp
        return normal

    def c1(self, r):
        r"""
        Determines the first Chern class corresponding to J_r, via

        .. math::
            \begin{align}
            c_1^r &= \bigg[ n_r +1 - \sum_{a=1}^{K} q_a^r \bigg].
            \end{align}
        
        Parameters
        ----------
        r : int
            the index of J_r.
        
        Returns
        -------
        c1: int
            The first Chern class corresponding to J_r.

        See Also
        --------
        c1: All first Chern classes.
        c2: Second Chern class of J_r, J_s.
        c3: Third Chern class of J_r, J_s, J_t.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.c1(0)
        0
        """
        c1 = self.M[r][0]+1-np.sum(self.M[r,1:])
        return c1

    def first_chern(self):
        r"""
        Determines c_1(M).

        .. math::
            \begin{align}
            c_1^r &= \bigg[ n_r +1 - \sum_{a=1}^{K} q_a^r \bigg].
            \end{align}
        
        Returns
        -------
        vector: array[nProj]
            First Chern class of M.

        See Also
        --------
        second_chern: Second Chern class.
        third_chern: Third Chern class.
        fourth_chern: Fourth Chern class.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.first_chern()
        [0,0]
        """
        vector = np.array([self.c1(i) for i in range(self.len)])
        return vector

    def c2(self, r, s):
        r"""
        Determines the second Chern class corresponding to J_r, J_s, via

        .. math::
            \begin{align}
            c_2^{rs}  =\frac{1}{2} \bigg[ - \delta^{rs} (n_r + 1 ) + \sum_{a=1}^{K} q_a^r q_a^s \bigg].
            \end{align}
        
        Parameters
        ----------
        r : int
            the index of J_r.
        s : int
            the index of J_s.
        
        Returns
        -------
        c2: float
            The second Chern class corresponding to J_r, J_s.

        See Also
        --------
        c1: First Chern class of J_r.
        c2: Second Chern class in vector notation.
        c3: Third Chern class of J_r, J_s, J_t.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.c2(0,1)
        2.5
        """
        sumqq = np.sum([self.M[r][i]*self.M[s][i] for i in range(1,self.K+1)], dtype=np.float32)
        if r==s:
            delta = self.M[r][0]+1
        else:
            delta = 0
        c2 = (sumqq-delta)/2.0 #Calabi-Yau
        if not self.CY:
            c2 += self.c1(r)*self.c1(s)/2.0 #non Calabi-Yau
        return c2

    def second_chern_all(self):
        r"""
        Determines all second Chern classes.

        .. math::
            \begin{align}
            c_2^{rs}  =\frac{1}{2} \bigg[ - \delta^{rs} (n_r + 1 ) + \sum_{a=1}^{K} q_a^r q_a^s \bigg].
            \end{align}
        
        Returns
        -------
        matrix: array[nProj, nProj]
            The full second Chern class.

        See Also
        --------
        first_chern: First Chern class.
        second_chern: Second Chern class in vector notation.
        third_chern: Third Chern class.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.second_chern_all()
        [[1.0, 2.5], [2.5, 3.0]]
        """
        matrix = np.array([[self.c2(i,j) for j in range(self.len)] for i in range(self.len)])
        return matrix

    def c3(self,r,s,t):
        r"""
        Determines the third Chern class corresponding to J_r, J_s, J_t, via

        .. math::
            \begin{align}
            c_3^{rst}  = \frac{1}{3} \bigg[ \delta^{rst} (n_r + 1 ) - \sum_{a=1}^{K} q_a^r q_a^s q_a^t \bigg].
            \end{align}
        
        Parameters
        ----------
        r : int
            the index of J_r.
        s : int
            the index of J_s.
        t : int
            the index of J_t.
        
        Returns
        -------
        c3: float
            The third Chern class corresponding to J_r, J_s, J_t.

        See Also
        --------
        c1: First Chern class of J_r.
        c2: Second Chern class of J_r, J_s.
        third_chern: Complete third Chern class.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.c3(0,1,1)
        -3.6666666666666665
        """
        sumqqq = np.sum([self.M[r][i]*self.M[s][i]*self.M[t][i] for i in range(1,self.K+1)], dtype=np.float32)
        if r==s and r==t:
            delta = self.M[r][0]+1
        else:
            delta = 0
        c3 = (delta-sumqqq)/3.0 # Calabi Yau
        if not self.CY:
            # for non CY
            c3 += self.c1(r)*self.c2(s,t)-self.c1(r)*self.c1(s)*self.c1(t)/3.0
        return c3
        
    def third_chern(self):
        r"""
        Determines c_3(M).

        .. math::
            \begin{align}
            c_3^{rst}  = \frac{1}{3} \bigg[ \delta^{rst} (n_r + 1 ) - \sum_{a=1}^{K} q_a^r q_a^s q_a^t \bigg].
            \end{align}
        
        Returns
        -------
        matrix: array[nProj, nProj, nProj]
            The full third Chern class.

        See Also
        --------
        first_chern: First Chern class.
        second_chern: Second Chern class in vector notation.
        third_chern: Third Chern class.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.c3()
        [[[-2.0, -2.3333333333333335], [-2.3333333333333335, -3.6666666666666665]], [[-2.3333333333333335, -3.6666666666666665], [-3.6666666666666665, -8.0]]]
        """
        matrix = np.array([[[self.c3(i,j,k) for k in range(self.len)] for j in range(self.len)] for i in range(self.len)])
        return matrix

    def c4(self, r, s, t, u):
        r"""
        Determines the fourth Chern class J_r, J_s, J_t, J_u for Calabi Yau four folds

        .. math::
            \begin{align}
            c_4^{rstu}  = \frac{1}{4} \bigg[ - \delta^{rstu} (n_r + 1 ) + \sum_{a=1}^{K} q_a^r q_a^s q_a^t q_a^u + 2 c_2^{rs} c_2^{tu} \bigg].
            \end{align}
        
        Parameters
        ----------
        r : int
            the index of J_r.
        s : int
            the index of J_s.
        t : int
            the index of J_t.
        u : int
            the index of J_u.
        
        Returns
        -------
        c4: float
            The fourth Chern class corresponding to J_r, J_s, J_t, J_u.

        See Also
        --------
        c1: First Chern class of J_r.
        c2: Second Chern class of J_r, J_s.
        c3: Third Chern class of J_r, J_s, J_t.
        fourth_chern: All fourth Chern classes.

        Example
        -------
        >>> M = CICY([[2,3],[2,3],[1,2]])
        >>> M.c4(0,1,1,2)
        20.25

        References
        ----------
        .. [1] All CICY four-folds, by J. Gray, A. Haupt and A. Lukas.
            https://arxiv.org/pdf/1303.1832.pdf
        """

        if self.nfold == 3 or self.nfold == 2:
            logger.error('{} is a Calabai Yau {}-fold.'.format(self.M, self.nfold))
        
        sumqqqq = np.sum([self.M[r][i]*self.M[s][i]*self.M[t][i]*self.M[u][i] for i in range(1,self.K+1)], dtype=np.float32)
        if r==s and r==t and r==u:
            delta = self.M[r][0]+1
        else:
            delta = 0
        second = 2.0*self.c2(r,s)*self.c2(t,u)
        c4 = (sumqqqq+second-delta)/4.0
        if not self.CY:
            c4 += 1/2.0*self.c2(r,s)*self.c2(t,u)+self.c1(r)*self.c3(s,t,u)-self.c1(r)*self.c1(s)*self.c2(t,u)+1/4.0*self.c1(r)*self.c1(s)*self.c1(t)*self.c1(u)
        return c4

    def fourth_chern(self):
        r"""
        Determines c_4(M).
        
        .. math::
            \begin{align}
            c_4^{rstu}  = \frac{1}{4} \bigg[ - \delta^{rstu} (n_r + 1 ) + \sum_{a=1}^{K} q_a^r q_a^s q_a^t q_a^u + 2 c_2^{rs} c_2^{tu} \bigg].
            \end{align}

        Returns
        -------
        matrix: array[nProj, nProj, nProj, nProj]
            The full fourth Chern class.

        See Also
        --------
        c1: First Chern class of J_r.
        c2: Second Chern class of J_r, J_s.
        c3: Third Chern class of J_r, J_s, J_t.
        c4: Foruth Chern class of J_r, J_s, J_t, J_u.

        Example
        -------
        >>> M = CICY([[2,3],[2,3],[1,2]])
        >>> M.fourth_chern()
        [[[[24.0, 27.0, 18.0], [27.0, 24.75, 18.0], [18.0, 18.0, 10.5]], ... , [[10.5, 11.25, 7.5], [11.25, 10.5, 7.5], [7.5, 7.5, 4.0]]]]
        """
        matrix = np.array([[[[self.c4(i,j,k,l) for l in range(self.len)]
                     for k in range(self.len)] for j in range(self.len)] for i in range(self.len)])
        return matrix

    def drst(self,r,s,t,x=1):
        r"""
        Determines the triple intersection number d_rst.
        We use:

        .. math::
            \begin{align}
             d_{rst} = \int_X J_r \wedge J_s \wedge J_t = \int_A \mu \wedge J_r \wedge J_s \wedge J_t 
            \end{align}

        where \mu is the top form
        
        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}

        Parameters
        ----------
        r : int
            index r.
        s : int
            index s.
        t : int 
            index t.
        x : int, optional
            Normalization for integral, by default 1.
        
        Returns
        -------
        drst: float
            Returns the triple intersection number drst.
    
        See also
        --------
        triple_intersection: Determines all triple intersection numbers.
        second_chern: The second Chern class as a vector.
        euler_characteristic: The euler_characteristicharacteristic.
        quadruple_intersection: The quadruple intersection numbers for a four fold.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.drst(0,1,1)
        7.0
        """
        if self.triple.shape[0] != 0:
            return self.triple[r][s][t]
        drst=0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination = np.array([0 for _ in range(self.K)])
        count = [0 for _ in range(self.len)]
        #now there are 5 distinct cases:
        #1) r=s=t or 2) all neqal or the 2-5) three cases where two are equal
        #1)
        if r==s==t:
            if self.M[r][0] < 3:
                #then drst is zero
                return 0
            else:
                i=0
                #now we want to fill combination and run over all m Projective spaces,
                # and how often they occur
                for j in range(self.len):
                    if j==r:
                        #we obviously have to subtract 3 in the case of three
                        # times the same index since we already have three kähler forms
                        # in Ambient space coming from the intersection number
                        count[j] = self.M[j][0]-3
                        combination[i:i+count[j]] = j
                        i += self.M[j][0]-3
                    else:
                        count[j] = self.M[j][0]
                        combination[i:i+count[j]] = j
                        i += self.M[j][0]
        # 2)                             
        if r!=s and r!=t and s!=t:
            i=0
            for j in range(self.len):
                if j==r or j==s or j==t:
                    count[j] = self.M[j][0]-1
                    combination[i:i+count[j]] = j
                    i += self.M[j][0]-1
                else:
                    count[j] = self.M[j][0]
                    combination[i:i+count[j]] = j
                    i += self.M[j][0]
        # 3)
        if r==s and r!=t:
            if self.M[r][0] < 2:
                return 0
            else:
                i=0
                for j in range(self.len):
                    if j==r:
                        count[j] = self.M[j][0]-2
                        combination[i:i+count[j]] = j
                        i += self.M[j][0]-2
                    else:
                        if j==t:
                            count[j] = self.M[j][0]-1
                            combination[i:i+count[j]] = j
                            i += self.M[j][0]-1
                        else:
                            count[j] = self.M[j][0]
                            combination[i:i+count[j]] = j
                            i += self.M[j][0]
        # 4)
        if r==t and r!=s:
            i=0
            if self.M[r][0] < 2:
                return 0
            else:
                i=0
                for j in range(self.len):
                    if j==r:
                        count[j] = self.M[j][0]-2
                        combination[i:i+count[j]] = j
                        i += self.M[j][0]-2
                    else:
                        if j==s:
                            count[j] = self.M[j][0]-1
                            combination[i:i+count[j]] = j
                            i += self.M[j][0]-1
                        else:
                            count[j] = self.M[j][0]
                            combination[i:i+count[j]] = j
                            i += self.M[j][0]
        # 5)
        if s==t and s!=r:
            i=0
            if self.M[s][0] < 2:
                return 0
            else:
                i=0
                for j in range(self.len):
                    if j==s:
                        count[j] = self.M[j][0]-2
                        combination[i:i+count[j]] = j
                        i += self.M[j][0]-2
                    else:
                        if j==r:
                            count[j] = self.M[j][0]-1
                            combination[i:i+count[j]] = j
                            i += self.M[j][0]-1
                        else:
                            count[j] = self.M[j][0]
                            combination[i:i+count[j]] = j
                            i += self.M[j][0]
        # the combinations of mu grow exponentially with self.K and the number of ambient spaces
        # Check, when the number of multiset_permutations become to large to handle
        if self.K < 8 and len(np.unique(combination)) < 6:
            # Hence, for large K and small self.len, this might take really long. 
            mu = sp.utilities.iterables.multiset_permutations(combination)
            # self.K!/(#x_1!*...*#x_n!)
            for a in mu:
                v = x
                for j in range(self.K):
                    if self.M[a[j]][j+1] == 0:
                        v = 0
                        break
                    else:
                        v *= self.M[a[j]][j+1]
                drst += v
            return drst
        else:
            # here we calculate the nonzero paths through the CICY
            # much faster since CICYs with large K and large self.len tend to 
            # be pretty sparse
            nonzero = [[] for _ in range(self.K)]
            combination = np.sort(combination)
            count_2 = [0 for _ in range(self.len)]
            # run over all K to find possible paths
            for i in range(self.K):
                for j in range(self.len):
                    # possible paths are non zero and in combination
                    if self.M[j][i+1] != 0 and j in combination:
                        nonzero[i] += [j]
                        count_2[j] += 1
            # Next we run over all entries in count to see if any are fixed by number of occurence
            for i in range(self.len):
                if count[i] == count_2[i]:
                    # if equal we run over all entries in nonzero
                    #count[i] = 0
                    for j in range(self.K):
                        # and fix them to i if they contain it 
                        if i in nonzero[j]:
                            # and len(nonzero[j]) != 1
                            nonzero[j] = [i]
            #There are some improvements here:
            #1) take the counts -= 1 if fixed and compare if the left allowed
            #2) here it would be even more efficient to write a product that respects 
            #   the allowed combinations from count.
            mu = itertools.product(*nonzero)
            # len(nonzero[0])*...*len(nonzero[K])
            # since we in principle know the complexity of both calculations
            # one could also do all the stuff before and then decide which way is faster
            for a in mu:
                # if allowed by count
                c = list(a)
                if np.array_equal(np.sort(c), combination):
                    v = x
                    for j in range(self.K):
                        if self.M[c[j]][j+1] == 0:
                            break
                        else:
                            v *= self.M[c[j]][j+1]
                    drst += v
            return drst

    def triple_intersection(self):
        """
        Determines all triple intersection numbers.
        
        Returns
        -------
        d: array[nProj, nProj, nProj]
            numpy array of all triple intersection numbers, d_rst.
        
        See also
        --------
        second_chern: The second Chern class as a vector.
        euler_characteristic: The eulercharacteristic.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.triple_intersection()
        array([[[0., 3.],
                [3., 7.]],
               [[3., 7.],
                [7., 2.]]])
        """
        if self.nfold != 3:
            logger.warning('CICY is not a 3-fold.')
        if self.triple.shape[0] != 0:
            return self.triple
        # Since the calculation of drst for large K becomes very tedious
        # we make use of symmetries
        comb = itertools.combinations_with_replacement(range(self.len), 3)
        d = np.zeros((self.len, self.len, self.len))
        for x in comb:
            drst = self.drst(x[0], x[1], x[2])
            entries = itertools.permutations(x, 3)
            # there will be some redundant elements,
            # but they will only have to be calculated once.
            for b in entries:
                d[b] = drst
        self.triple = d
        return d

    def second_chern(self):
        r"""
        Uses the triple intersection numbers to contract the second chern matrix to a vector:

        .. math::
            \begin{align}
             c_{2;t} = d_{rst} c_2^{rs}.
            \end{align}
        
        Returns
        -------
        chern: array[nProj]
            The second Chern class as a vector.

        See also
        --------
        triple_intersection: Determines the triple intersection numbers.
        second_chern_all: The second Chern class as a matrix.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.c2()
        [36.0, 44.0]
        """
        c2 = self.second_chern_all()
        chern = np.einsum('rst,st -> r', self.triple, c2)
        return chern

    def drstu(self,r,s,t,u,x=1):
        r"""
        Determines the quadruple intersection numbers, d_rstu, for Calabi Yau 4-folds.
        
        Parameters
        ----------
        r : int
            the index r.
        s : int
            the index s.
        t : int
            the index t.
        u : int
            the index u.
        x : int, optional
            Normalization for integral, by default 1.

        Returns
        -------
        drstu: float
            The quadruple intersection number d_rstu.

        See Also
        --------
        drst: Determines the triple intersection number d_rst.
        euler_characteristic: The eulercharacteristic.
        quadruple_intersection: All quadruple intersection numbers of a 4-fold.

        Example
        -------
        >>> M = CICY([[2,3],[2,3],[1,2]])
        >>> M.drstu(0,1,1,2)
        3

        References
        ----------
        .. [1] All CICY four-folds, by J. Gray, A. Haupt and A. Lukas.
            https://arxiv.org/pdf/1303.1832.pdf
        """

        if self.nfold != 4:
            logger.warning('CICY is not a 4-fold.')

        if self.quadruple.shape[0] != 0:
            return self.quadruple[r][s][t][u]
        drstu=0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination = np.array([0 for _ in range(self.K)])
        count = [0 for _ in range(self.len)]
        #now there are 5 distinct cases:
        #1) r=s=t=u or 2) all neqal or the 3) two equal, two nonequal
        #4) two equal and two equal 5) three equal
        un, unc = np.unique([r,s,t,u], return_counts=True)
        for i in range(len(un)):
            if self.M[un[i]][0] < unc[i]:
                return 0
        i = 0
        for j in range(self.len):
            #if j in rstu subtract
            # else go full
            contained = False
            for a in range(len(un)):
                if j == un[a]:
                    contained = True
                    count[j] = self.M[j][0]-unc[a]
                    combination[i:i+count[j]] = j
                    i += self.M[j][0]-unc[a]
            if not contained:
                count[j] = self.M[j][0]
                combination[i:i+count[j]] = j
                i += self.M[j][0]
        # just copy from drst
        # the combinations of mu grow exponentially with self.K and the number of ambient spaces
        # Check, when the number of multiset_permutations become to large to handle
        if self.K < 8 and len(np.unique(combination)) < 6:
            # Hence, for large K and small, this might take really long. 
            mu = sp.utilities.iterables.multiset_permutations(combination)
            # self.K!/(#x_1!*...*#x_n!)
            for a in mu:
                v = x
                for j in range(self.K):
                    if self.M[a[j]][j+1] == 0:
                        v = 0
                        break
                    else:
                        v *= self.M[a[j]][j+1]
                drstu += v
            return drstu
        else:
            # here we calculate the nonzero paths through the CICY
            nonzero = [[] for _ in range(self.K)]
            combination = np.sort(combination)
            count_2 = [0 for _ in range(self.len)]
            # run over all K to find possible paths
            for i in range(self.K):
                for j in range(self.len):
                    # possible paths are non zero and in combination
                    if self.M[j][i+1] != 0 and j in combination:
                        nonzero[i] += [j]
                        count_2[j] += 1
            # Next we run over all entries in count to see if any are fixed by number of occurence
            for i in range(self.len):
                if count[i] == count_2[i]:
                    # if equal we run over all entries in nonzero
                    #count[i] = 0
                    for j in range(self.K):
                        # and fix them to i if they contain it 
                        if i in nonzero[j]:
                            # and len(nonzero[j]) != 1
                            nonzero[j] = [i]
            #There are some improvements here:
            #1) take the counts -= 1 if fixed and compare if the left allowed
            #2) here it would be even more efficient to write a product that respects 
            #   the allowed combinations from count, but I can't be bothered to do it atm.
            mu = itertools.product(*nonzero)
            # len(nonzero[0])*...*len(nonzero[K])
            # since we in principle know the complexity here and from the other
            # one should also do all the stuff before and then decide which way is faster
            for a in mu:
                # if allowed by count
                c = list(a)
                if np.array_equal(np.sort(c), combination):
                    v = x
                    for j in range(self.K):
                        if self.M[c[j]][j+1] == 0:
                            break
                        else:
                            v *= self.M[c[j]][j+1]
                    drstu += v
            return drstu

    def quadruple_intersection(self):
        """
        Determines all quadruple intersection numbers.
        
        Returns
        -------
        d: array[nProj, nProj, nProj, nProj]
            numpy array of all quadruple intersection numbers, d_rstu.
        
        See also
        --------
        triple_intersection: The triple intersection numbers of a 3-fold.
        euler_characteristic: The eulercharacteristic.

        Example
        -------
        >>> M = CICY([[2,3],[2,3],[1,2]])
        >>> M.quadruple_intersection(0,1,1,2)
        array([[[[0., 0., 0.],
                 [0., 2., 3.],
                 [0., 3., 0.]],
                    ...
                [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.]]]])
        """
        # check if they have been calculated before
        if self.quadruple.shape[0] != 0:
            return self.quadruple
        # Since the calculation of drstu for large K becomes very tedious
        # we make use of symmetries
        comb = itertools.combinations_with_replacement(range(self.len), 4)
        d = np.zeros((self.len, self.len, self.len, self.len))
        for x in comb:
            drstu = self.drstu(x[0], x[1], x[2], x[3])
            entries = itertools.permutations(x, 4)
            # there will be some redundant elements,
            # but they will only have to be calculated once.
            for b in entries:
                d[b] = drstu
        self.quadruple = d
        return d

    def euler_characteristic(self):
        r"""
        Determines the Euler characteristic via integration of the Chern class. Take e.g. n=3

        .. math::
            \begin{align}
             \chi = \frac{1}{2} \int_X c_3 \; .
            \end{align}
        
        Returns
        -------
        e: float
            The euler characteristic.

        See also
        --------
        drst: Determines the triple intersection number d_rst.
        c3: The third Chern class of J_r, J_s, J_t.
        drstu: The quadruple intersection number d_rstu for a four fold.
        c4: The fourth Chern class of J_r, J_s, J_t, J_u.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.euler_characteristic()
        -114.0
        """
        if self.euler != 1:
            return self.euler
        e = 0
        if self.nfold == 3:
            d = self.triple_intersection()
            c3 = self.third_chern()
            e =  np.einsum('rst,rst', d, c3)
        elif self.nfold == 4:
            d = self.quadruple_intersection()
            c4 = self.fourth_chern()
            e =  np.einsum('rstu,rstu', d, c4)
        elif self.nfold == 2:
            e = 24
        self.euler = e
        #int(e) makes 0.9999 float to 0 which comes from the third Chern,
        # hence we use round
        return round(e)

    def _fill_moduli(self, seed):
        """Determines a tuple with monomials and their (redundant) complex moduli
             as coefficient for the defining Normal sections.
        
        Args:
            seed (int): Random seed for the coefficients
        
        Returns:
            tuple 
                1: (nested list: int): nested list of all monomials
                2: (nested list: int): nested list of all coefficients
        """
        random.seed(seed)
        # find the sections of the normal
        sec_norm = [[self.M[i][j] for i in range(self.len)] for j in range(1, self.K+1)]
        # declare return lists
        pmoduli = [0 for i in range(self.K)]
        moduli = [0 for i in range(self.K)]
        #fill them
        for i in range(self.K):
            dim = int(self._brackets_dim(sec_norm[i]))
            pmoduli[i] = self._makepoly(sec_norm[i], dim)
            # fill with random values, generic polynomials
            # could make it optional to give the range of the values
            moduli[i] = np.array([randint(1, 50) for j in range(dim)], dtype=np.int16)
        return pmoduli, moduli

    @staticmethod
    def _BBW(V):
        r"""Applies the BBW theorem to a Vector bundle in BBW notation.
           Schematic calculation follows:
                (1|-101) -> (2135) -> (1235) -> [[0,0,0,1],j=1]
           Returns the new vector notation and value for j = # of swaps.
        
        Args:
            V (nested list: int): A vector bundle in BBW notation, see example
        
        Returns:
            (nested list: int): Returns a list with transformed vector in BBW notation and non zero cohomology;
                                    or [0,500] in case of vanishing cohomology.

        Example:
            O(2) over P^1:
              >>> BBW([2,0])
              [[1,2], 1]
            or
              >>> BBW([1,-1,0,1])
              [[0,0,0,1],1]
        """
        # input + sequence
        vector = [V[i]+i for i in range(len(V))]
        j = 0
        # permutations
        save = [0 for i in range(len(V))]
        # Next we run over V and sort it according to BBW
        # essentially bubblesort?
        for _ in range(len(V)-1):
            for n in range(len(V)-1):
                # if two entries are the same -> return zero
                if vector[n] == vector[n+1]:
                    return [0, 500]
                else:
                    # if n > n+1 then we interchange and add +1 to j
                    if vector[n] > vector[n+1]:
                        save[n] = vector[n]
                        vector[n] = vector[n+1]
                        vector[n+1] = save[n]
                        j = j+1
        # Now that we have sorted the vector, we need to subtract the sequence
        save = [vector[i]-i for i in range(len(V))]
        return [save, j]

    def _hom(self, V, k):
        r"""
        Determines the nonzero entries in the first Leray instance.
           This is achieved by calculating all vector bundles resulting from
           \wedge^k N \otimes V .
           Returns then a list of all surviving representations and their origin.
        
        Args:
            V (nested list: int): Vector bundle in BBW notation
            k (int): \wedge^k k-th wedge product of the Normalbundle.
        
        Returns:
            tuple
                1. (nested list: int): [BBW notation after BBW, j]
                2. (nested list: int): list of integers indicating the Normal bundle origin of each entry

        Example:
            >>> M.hom([324], 2)
             
        """
        # We begin by finding the amount of possible combinations coming from k
        x = int(sc.special.binom(self.K, k))

        # Next, we create a new matrix for the wedge products
        matrix = [[0 for i in range(self.len)] for q in range(x)]
        # and the origins
        origin = [[0 for i in range(self.len)] for q in range(x)]

        # We fill the matrix with the #k=x combinations
        # Since the normal bundle consists of line bundles we only save the total
        # integer for each combination
        for a in range(self.len):
            # We create a list saving the combinations of N[a] as (A,B), (A,C),..
            combvector = list(itertools.combinations(self.N[a], k))
            originvector = list(itertools.combinations([l for l in range(self.K)], k))
            # Next we need to determine the (wedge) products of all these x combinations
            # For that we take the sum of all entries in combvector
            for b in range(x):
                for c in range(k):
                    matrix[b][a] = matrix[b][a]+combvector[b][c]
                origin[b][a] = originvector[b][:]
        
        # Next we need to transform the integer in a valid vector notation for
        # BBW to work with and add the vector which cohomology we are interested in
        for a in range(self.len):
            for b in range(x):
                # each integer becomes a vector of length corresponding projective
                # space + 1, where the integer is the first entry
                # Introduce save
                save = matrix[b][a]
                # add/substract the vector
                # Here the code made a ref, be careful
                matrix[b][a] = V[a][:]
                # add to the first element our saved integer
                matrix[b][a][0] = matrix[b][a][0]+save
        
        # We should now have a matrix with representations which we can feed into
        # BBW. BBW will then return, a cohomology and a value j, which are non zero
        Bott = [[0 for a in range(self.len)] for b in range(x)]
        j = [0 for b in range(x)]
        ob = [0 for b in range(x)]
        for b in range(x):
            for a in range(self.len):
                save = self._BBW(matrix[b][a])
                if save[1] == 500:
                    # This means that this whole [b] entry should be zero
                    Bott[b] = 0
                    # we need to end the [a] loop and go to the next [b]
                    j[b] = 500
                    break
                else:
                    # it is not zero we simply set it equal to representation from BBW
                    Bott[b][a] = save[0]
                    j[b] = j[b]+save[1]
                    # and save the origin
                    ob[b] = origin[b][a]
        # We return a vector of cohomologies for each possible product-vector
        return [Bott, j], ob

    def _brackets_dimv(self, B):
        r"""
        Determines the dimension of a vector bundle in bracket notation.
        
        Args:
            B (nested list: int): Vector bundle in bracket notation.
        
        Returns:
            (int): dimension of that Vector bundle.
        """
        dim = 1
        x = [[abs(b) for b in a] for a in B]
        # loop over all projective spaces
        for a in range(self.len):
            # loop over all 'tensors'
            for b in range(len(x[a])): 
                dim = dim*sc.special.binom(self.M[a][0]+x[a][b], self.M[a][0])
            if len(x[a]) == 2:
                #since traceless subtract 1
                dim -= 1
        return int(dim)

    def _vector_brackets(self, V):
        r"""
        Takes a vector(tangent ambient) bundle in BBW notation and
        transforms into Bracket notation.
        
        Follows schematically:
            (-1|01)        
            (-2|00 )  --> [[1,-1],[2],[2]]
            (-2|000)
        
        Args:
            V (nested list: int): A vector bundle in BBW notation
        
        Returns:
            (nested list: int): A vector bundle in bracket notation.
        """
        brackets = [[] for a in range(len(V))]
        # Now for line bundles there are two cases
        for i in range(len(V)):
            #If BBW flipped the dimension to the right,
            #  we have a 1 in first entry,
            if V[i][0] == 1:
                # then we know that the important entries sit on the right end
                # we iterate over all and save all which are != 1
                if V[i][-1] == 1:
                    brackets[i] +=[0]
                else:
                    for j in range(len(V[i])):
                        if V[i][j] > 1:
                            brackets[i] += [(-1)*(V[i][j]-1)]
            else:
                #if == 0 we have a 'scalar'
                if V[i][0] == 0:
                    if V[i][-1] == 0:
                        brackets[i] +=[0]
                    else:
                        for j in range(len(V[i])):
                            if V[i][j] > 0:
                                brackets[i] += [(-1)*(V[i][j])]
                else:
                    # otherwise run over all and check all nonzero entries
                    for j in range(len(V[i])):
                        if V[i][j] < 0:
                            brackets[i] += [0-V[i][0]]
                        if V[i][j] > 0:
                            brackets[i] += [(-1)*(V[i][j])]
        return brackets

    def Leray(self, V, line=True):
        r"""
        Determines the first instance, i=1, of a Leray table for a given vector bundle V.

        .. math::

            \begin{align}
	        E_{i+1}^{j,k} = \frac{\text{Ker}(d_i : E_{i}^{j,k} (\mathcal{V}) \rightarrow E_i^{j-1,k-1}(\mathcal{V}) )}{\text{Im}(d_i : E_{i}^{j+i-1,k+i} (\mathcal{V}) \rightarrow E_i^{j,k}(\mathcal{V}) )}
	        \end{align}
        
        V has to be in proper BBW notation, this is most easily achieved, by taking e.g. a line bundle L = [q_1 , ... ,q_n] and

        >>> V = M._line_to_BBW(L)

        Parameters
        ----------
        V : nested list
            A vector bundle in BBW notation.
        line : bool, optional
            True, if the vector bundle is a line bundle, by default True.
        
        Returns
        -------
        E: nested list
            The first Leray instance
        origin: nested list
            The origin of each non trivial entry.

        See also
        --------
        line_co: Line bundle cohomology of L.
        _line_to_BBW: Transforms a line bundle into BBW notation.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> V = M._line_to_BBW([3,-4])
        >>> M.Leray(V)
        ([[0, 0, 0, [[3, 0]], 0, 0], [0, 0, 0, [[1, -1], [2, -3]], 0, 0], [0, 0, 0, [[0, -4]], 0, 0]],
         [[0, 0, 0, 0, 0, 0], [0, 0, 0, [(0,), (1,)], 0, 0], [0, 0, 0, [(0, 1)], 0, 0]])
        """
        # Build Leray table E_1[k][j]
        E = [[0 for a in range(self.dimA+1)] for b in range(self.K+1)]
        # origin saves where the rep came from
        origin = [[0 for a in range(self.dimA+1)] for b in range(self.K+1)]
        # First we fill k=0 since our other algorithm makes some problems there;
        # this will always be only one cohomology entry/ stacked list
        E_0 = [self._BBW(V[a]) for a in range(self.len)]
        x = 0
        # we look if any of the stacked representations yields 0 --> 0
        # if not we find the appropriate cohomology j value
        for a in range(self.len):
            if E_0[a][1] == 500:
                x = 500
                break
            else:
                x = x + E_0[a][1]
        if x != 500:
            # if cohomology is non zero we put it at the right j value
            E[0][x] = [[E_0[0][0]]]
            #origin[0][x] = [0]
            for a in range(1, self.len):
                E[0][x][0].append(E_0[a][0])
            # make line bracket notation
            if line:
                E[0][x][0] = self._line_brackets(E[0][x][0])
            else:
                E[0][x][0] = self._vector_brackets(E[0][x][0])
        # if cohomology is zero we simply leave all j for k=0 as zero.
        # Next we fill the higher entries of k
        # create a list which checks if cohomology class is non zero and saves
        # the cohomology rep in 1st and j in 2nd entry
        # Table = [[0, 500]]
        # Now we run a loop over all k entries and fill the table
        for k in range(1, self.K+1):  # go over entries 1 to K
            Table, torigin = self._hom(V, k)  # fill table with cohomology values
            # now we fill our leray table with all found cohomologies
            for a in range(len(Table[0])):
                if Table[1][a] != 500:  #ask if cohomolgy changed, i.e. is non zero
                    x = int(Table[1][a])  # extract j of cohomolgy
                    # Now we want to add those non zero cohomologies to our table
                    if E[k][x] == 0:  # if it is the first non zero j, then equal
                        #E[k][x] = [Table[0][a]]# in BBW notation
                        if line:
                            E[k][x] = [self._line_brackets(Table[0][a])]
                        else:
                            E[k][x] = [self._vector_brackets(Table[0][a])]
                        origin[k][x] = [torigin[a]]
                    else:  # if there is already a nonzero entry, we append
                        #E[k][x].append(Table[0][a])# in BBW
                        if line:
                            E[k][x].append(self._line_brackets(Table[0][a]))
                        else:
                            E[k][x].append(self._vector_brackets(Table[0][a]))
                        origin[k][x].append(torigin[a])
        return E, origin

    def _line_to_BBW(self, L):
        r"""
        Transforms a line bundle into BBW notation.
        
        Args:
            L (list: int): line bundle
        
        Returns:
            (nested list: int): line bundle in BBW notation
        
        Example:
            Consider a CICY X with ambient space P^1, P^2, P^3
            >> X._line_to_BBW([1,2,3])
             [[-1,0],[-2,0,0],[-3,0,0,0]]
        """
        linebundle = [[(-1)*L[a]] for a in range(self.len)]
        # Need a loop which adds 0 according to dim(P_a)
        for a in range(self.len):
            linebundle[a] = linebundle[a] + [0 for i in range(self.M[a][0])]
        return linebundle

    def _line_brackets(self, V):
        r"""
        Takes a line bundle in BBW notation and transforms into bracket notation.
        
        Follows schematically:
            (-1|0  )
            (-2|00 )  --> [1,2,2]
            (-2|000)

        Args:
            V (nested list: int): A line bundle in BBW notation
        
        Returns:
            (list: int): A line bundle in bracket notation
        """
        brackets = [0 for a in range(len(V))]
        # Now for line bundles there are two cases
        for i in range(len(V)):
            #If BBW flipped the dimension to the right,
            #  we have a 1 in first entry,
            if V[i][0] == 1:
                # then we assign the following value/degree of polynomial
                brackets[i] = (-1)*(V[i][-1]-1)
            else:
                #if != 1 then we haven't flipped at all in BBW,
                #  and we keep the value
                brackets[i] = 0-V[i][0]
        return brackets

    def _brackets_dim(self, B):
        r"""
        Determines the dimension of a line bundle in bracket notation.
        
        Args:
            B (list: int): Line bundle in bracket notation.
        
        Returns:
            (int): dimension of that line bundle.
        """
        dim = 1
        x = [abs(a) for a in B]
        for a in range(self.len): 
            dim = dim*sc.special.binom(self.M[a][0]+x[a], self.M[a][0])
        return int(dim)

    def _lorigin(self, or1, or2):
        r"""
        Takes two origins and returns a list of allowed mappings.
        
        Args:
            or1 (list: int): Origins of the first bundle
            or2 (list: int): Origins of the second bundle
        
        Returns:
            (nested list): [[bool, int], ...] of allowed mappings and positions.
        """
        #run a loop over all entries in or2
        origin = [[False, 500] for a in range(self.K)]
        if or2 != 0:
            for i in range(len(or2)):
                if set(or2[i]).issubset(set(or1)):
                    origin[list(set(or1).difference(set(or2[i])))[0]] = [True, i]
        else:
            origin[or1[0]] = [True, 0]
        return origin

    def hodge_data(self):
        r"""
        Determines the hodge numbers of the CICY. Based on Euler and adjunction sequence.
        I checked the results for all three folds against the ones found in the CICYlist.
        The computation of the four fold hodge numbers, however, has only been checked for some selected examples.
        Hence, the results should be taken with care and compared to the literature.
        
        Returns
        -------
        h: array[nfold+1]
            hodge numbers of the CICY.

        See also
        --------
        euler_characteristic: Determines the euler characteristic

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.hodge_data()
        [0, 59, 2.0, 0]

        References
        ----------
        .. [1] CY - The Bestiary, T. Hubsch
            http://inspirehep.net/record/338506?ln=en
        .. [2] Topologial Invariants and Fibration Structure of CICY 4-folds
            J. Gray, A. Haupt, A. Lukas
            1405.2073
        """
        # first we check if direct product
        h = [0 for _ in range(self.nfold+1)]
        if self.K > 1:
            test, text = self.is_directproduct()
            if test:
                if self.doc:
                    logger.info('The configuration matrix corresponds to a direct product:')
                    logger.info(text)
                # Apply Kunneth to get proper hodge data?
                # we return zero list, to be consistent with the CICYlist
                return h

        # disable doc, since a lot of trivial line bundle computations
        # we are not particular interested in follow.
        documentation = np.copy(self.doc)
        self.doc = False

        # We begin with \mathcal{N}
        # dimensions of all defining hypersurfaces
        normal_dimensions = np.zeros((self.K, self.nfold+1))
        # The total dimensions
        n_dim = [0 for _ in range(self.nfold+1)]
        # we run over each defining hypersurface
        for i in range(self.K):
            normal_dimensions[i] = self.line_co(np.transpose(self.N)[i])
            for j in range(self.nfold+1):
                # add all dimension
                n_dim[j] += normal_dimensions[i][j]

        # Next \mathcal{R}
        # dimensions of all unit hypersurfaces
        unit_dimensions = np.zeros((self.len, self.nfold+1))
        # The total dimensions
        u_dim = [0 for _ in range(self.nfold+1)]
        # The total space
        for i in range(self.len):
            unit_dimensions[i] = self.line_co([0 if j != i else 1 for j in range(self.len)])
            for j in range(self.nfold+1):
                # add up to the dimensions
                u_dim[j] += (self.M[i][0]+1)*unit_dimensions[i][j]

        # enable doc again.
        self.doc = documentation
        
        if self.nfold == 3:

            # we only need to determine h^21 or h^11 since they
            # are related via Euler characteristic
            # need to calculate kernel(H^1(X,S) -> H^1(X,N))
            if n_dim[1] == 0:
                kernel = u_dim[1]
            elif u_dim[1] == 0:
                kernel = 0
            else:
                # generic surjective map, matches the results for all CICY threefolds in the literature
                # We don't really need the space then.
                kernel = np.max([0, u_dim[1]-n_dim[1]])

            if self.doc:
                logger.info('We find the following dimensions in the long exact cohomology sequence of')
                logger.info('T_X     -> T_A|_X   -> N \n ----------------------------')
                logger.info('0       -> '+'{0: <8}'.format(str(u_dim[0]-self.len))+' -> '+str(n_dim[0]))
                logger.info('h^{2,1} -> '+'{0: <8}'.format(str(u_dim[1]))+' -> '+str(n_dim[1]))
                kernel222 = np.max([0, float(self.len)-u_dim[3]])
                logger.info('h^{1,1} -> '+'{0: <8}'.format(str(u_dim[2]+kernel222))+' -> '+str(n_dim[2]))
                logger.info('0       -> '+'{0: <8}'.format(str(u_dim[3]))+' -> '+str(n_dim[3]))

            # fill in h^21=h1 and h^11=h2
            h[1] = n_dim[0]-u_dim[0]+self.len+kernel
            h[2] = self.euler_characteristic()/2+h[1]
            return h
        elif self.nfold == 4:
            # first we check if direct product
            logger.warning('Hodge numbers have only been checked for all 3-folds. \n'+
                            'Double check your results with the literature.')

            if self.doc:
                logger.info('We find the following dimensions in the long exact cohomology sequence:')
                logger.info('T_X     -> T_A|_X   -> N \n ----------------------------')
                logger.info('0       -> '+'{0: <8}'.format(str(u_dim[0]-self.len))+' -> '+str(n_dim[0]))
                logger.info('h^{3,1} -> '+'{0: <8}'.format(str(u_dim[1]))+' -> '+str(n_dim[1]))
                logger.info('h^{2,1} -> '+'{0: <8}'.format(str(u_dim[2]))+' -> '+str(n_dim[2]))
                kernel222 = np.max([0, float(self.len)-u_dim[4]])
                logger.info('h^{1,1} -> '+'{0: <8}'.format(str(u_dim[3]+kernel222))+' -> '+str(n_dim[3]))
                logger.info('0 -> '+'{0: <8}'.format(str(u_dim[4]))+' -> '+str(n_dim[4]))

            # need to calculate kernel(H^1(X,S) -> H^1(X,N))
            if n_dim[1] == 0:
                kernel = u_dim[1]
            elif u_dim[1] == 0:
                kernel = 0
            else:
                # surjective as for threefold
                kernel = np.max([0, u_dim[1]-n_dim[1]])
            # fill in h^31=h1 and h^21=h2 and h^11 = h3 and 
            # with some abuse of notation we redefine h4 := h^22 
            h[1] = n_dim[0]-u_dim[0]+self.len+kernel
            # check if the sequence splits anywhere
            if u_dim[2] == 0:
                h[2] = n_dim[1]-(u_dim[1]-kernel)
                h[3] = self.euler_characteristic()/6+h[2]-h[1]-8
                h[4] = 44+4*h[3]-2*h[2]+4*h[1]
            elif n_dim[2] == 0:
                h[2] = n_dim[1]-(u_dim[1]-kernel)+u_dim[2]
                h[3] = self.euler_characteristic()/6+h[2]-h[1]-8
                h[4] = 44+4*h[3]-2*h[2]+4*h[1]
            else:
                # assume surjectiv again
                h[2] = n_dim[1]-(u_dim[1]-kernel)+np.max([0, u_dim[2]-n_dim[2]])
                h[3] = self.euler_characteristic()/6+h[2]-h[1]-8
                h[4] = 44+4*h[3]-2*h[2]+4*h[1]
            return h
        elif self.nfold == 2:
            # K3
            return [0, 20, 0]
        else:
            logger.error('Hodge calculation is only implemented for n=2,3,4 CY folds and'+
                            ' properly supported for 3 folds.')

    def is_favourable(self):
        """
        Determines if the CICY is favourable, i.e.
        h^{1,1} = number of projective spaces.
        
        Returns
        -------
        self.fav: bool
            True for favourable CICYs, False for non.

        See also
        --------
        is_directproduct: Determines if the CICY is a direct product.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.is_favourable()
        True
        """
        return self.fav

    def is_directproduct(self):
        """
        Determines if a CICY is a direct product.
        
        Returns
        -------
        direct: bool
            True if direct product, else False.
        product: list
            If direct == False, contains the components of the direct product and their position,
            else an empty list.

        See also
        --------
        is_favourable: Determines if the CICY is favourable.

        Examples
        --------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.is_directproduct()
        (False, []) 
        >>> D = CICY([[2,3,0],[3,0,4]])
        >>> D.is_directproduct()
        (True, [['T', [0]], ['K3', [1]]])
        """
        direct = False
        trans = np.transpose(self.N)
        product = []
        #1d check is easy
        for i in range(self.len):
            x = np.argmax(self.N[i])
            if self.N[i][x] == sum(self.N[i]):
                if trans[x][i] == sum(trans[x]):
                    # then there is a direct product
                    direct = True
                    if trans[x][i] == 3:
                        # Torus
                        if self.M[i][0] == 2:
                            product += [['T', [x]]]
                        else:
                            logger.warning('Configuration matrix is not Calabi Yau: '+str([i,x]))
                    if trans[x][i] == 2:
                        direct = False
                        # redundant or wrong CICY
                        if self.M[i][0] == 1:
                            logger.warning('The CICY is redundant here: '+str([i,x]))
                        else:
                            logger.warning('Configuration matrix is not Calabi Yau: '+str([i,x]))
                    if trans[x][i] == 4:
                        # K3
                        if self.M[i][0] == 3:
                            product += [['K3', [x]]]
                            # if three fold we found K3xT
                            if self.nfold == 3:
                                return direct, product
                        else:
                            logger.warning('Configuration matrix is not Calabi Yau: '+str([i,x]))
                    if trans[x][i] == 5:
                        # Quintic
                        if self.M[i][0] == 4:
                            product += [['Quintic', [x]]]
                            # if three fold we found Qunitic, else Quintic times Torus
                            if self.nfold == 4:
                                return direct, product
                            if self.nfold == 3:
                                direct = False
                                logger.warning('Are you the Quintic?')
                                return direct, product
                        else:
                            logger.warning('Configuration matrix is not Calabi Yau: '+str([i,x]))
        """
        possible time efficiency improvement here
        # first we check if we found any products, then we should investigate the other factor
        if product != []:
            return direct, product
        """
        """
        # in principle we should check whether self.K or self.len is bigger
        # so that we work with less combinations. The implementation of both
        # is however slightly differnt and we stick to self.K, i.e. the normal sections.
        # check whether it is easier to work with transpose or normal
        if self.K > self.len:
            l = self.K
            conf = np.sign(self.N)
        else:
            l = self.len
            conf = np.sign(trans)
        """
        #conf = np.sign(self.N)
        #loop over all combinations; start at i=1
        for i in range(1, int(self.K/2)+2):
            if self.K == 2 and i > 1:
                return direct, product
            # be careful this can also become quite large
            combs = list(itertools.combinations([k for k in range(self.K)], i))
            for j in range(len(combs)):
                x1 = np.sum(self.N[:, combs[j]], axis=1)
                x2 = np.sign(np.subtract(np.sum(self.N, axis=1), x1))
                x1 = np.sign(x1)
                if np.array_equal(x2+x1, np.ones((self.len))):
                    #what product are we looking at?
                    direct =  True
                    dim = sum([self.M[k][0] for k in range(self.len) if x1[k] == 1])-i
                    if dim == 1:
                        product += [['T', combs[j]]]
                    if dim == 2:
                        product += [['K3', combs[j]]]
                    if dim == 3:
                        product += [['CY3', combs[j]]]
        # the returned product list might be redundant as TxT will also yield a 'K3', etc.
        return direct, product

    def _single_map(self, V1, dim_V1, V2, dim_V2, t):
        """Determine the matrix of shape (dim_V2,dim_V1) for the map from V1 to V2.
        
        Args:
            V1 (list: int): Line bundle in bracket notation
            dim_V1 (int): dimension of V1
            V2 (list: in): Line bundle in bracket notation
            dim_V2 (int): dimension of V2
            t (int): specifies the normal section used for the map
        
        Returns:
            (nested list: int): A matrix for the map between the two monomial basis.
                with entries being the corresponding complex modulis.
        """

        V2 = [abs(entry) for entry in V2]
        V1 = [abs(entry) for entry in V1]
        #smatrix = np.zeros((dim_V1, dim_V2), dtype=np.int16)
        smatrix = np.zeros((dim_V2, dim_V1), dtype=np.int32)
        source = self._makepoly(V1, dim_V1) #only consider positive exponents
        moduli = np.subtract(V2, V1) # the modulimaps can contain derivatives
        dim_mod = self._brackets_dim(moduli)
        mod_polys = self._makepoly(moduli, dim_mod)
        v2poly = self._makepoly(V2, dim_V2)
        # loop over all monomials

        if self.doc:
            start = time.time()

        for i in range(dim_V1):
            # loop over all moduli 'monomials'
            for j in range(dim_mod):
                monomial = []
                #derivative = 1
                # determine the new monomial
                for x,y in zip(source[i], mod_polys[j]):
                    # if y is bigger we simply multiply
                    if y >= 0:
                        monomial += [x+y]
                    else:
                        # if smaller we see if taking the partial derivative yields zero
                        if abs(y) > x:
                            monomial = []
                            break
                        else:
                            # x is of sufficient power we save the factors coming from the derivative
                            # and lower the exponent appropiately
                            monomial += [x+y]
                            # apparently the derivative has to be taken without the prefactors.
                            # This is justified since, when there is a -1 we should rather imagine
                            # the map 1/mono -> 1/mono which is not actually a derivative.
                            #derivative *= int(np.product([z for z in range(x,x+y,-1)]))
                if monomial != []:
                    #brute force approach 
                    #for k in range(len(v2poly)):
                    #    if np.array_equal(monomial,v2poly[k]):
                    #        smatrix[k][i] = self.moduli[t][j]*derivative
                    #        #smatrix[i][k] = self.moduli[t][j]*derivative
                    #        break
                    #np approach; factor of ~150 faster
                    k = np.where(np.all(monomial == v2poly, axis=1))[0][0]
                    smatrix[k][i] = self.moduli[t][j]#*derivative
                    #nicer approach which finds the position; not implemented yet.
                    #pos = self.decoder(monomial, V2, dim_V2)
                    #smatrix[i][pos] = self.moduli[t][j]*derivative
        #print(smatrix)

        if self.doc:
            end = time.time()
            logger.debug('Time: {}'.format(end-start))

        return smatrix

    def _rank_map(self, V1, V2, V1o, V2o, SpaSM=False):
        """Determines the rank of a map between two Leray entries.
        The function creates a matrix of shape(dim_v1,dim_v2)

        Args:
            V1 (nested list: int): list of vector notations from which we map
            V2 (nested list: int): list of vector notations to which we map
            V1o (nested list: int): list of origins of all vectors in V1
            V2o (nested list: int): list of origins of all vectors in V2
            SpaSM (bool): If True: uses SpaSM library, default False
        
        Returns:
            (list: int): dimensions of [kernel, image] of the map; image=rank of the matrix
        """

        # Check if V1 or V2 are zero then trivial
        if V1 == 0:
            return [0,0]
        else:
            if V2 == 0:
                dimension = np.array([self._brackets_dim(vector) for vector in V1])
                return [np.sum(dimension), 0]

        # We fill the first list, V1, with bracket notation
        dim_bracket_V1 = [self._brackets_dim(V1[j]) for j in range(len(V1))]
        dim_V1 = sum(dim_bracket_V1)

        # We fill the second list, V2, with bracket notation
        dim_bracket_V2 = [self._brackets_dim(V2[j]) for j in range(len(V2))]
        dim_V2 = sum(dim_bracket_V2)

        if self.doc:
            logger.info('We determine the map from \n {} \n to \n {} \n with dimensions {} and {}.'.format(np.array(V1), np.array(V2),
                             [dim_bracket_V1, dim_V1], [dim_bracket_V2,dim_V2]))
            f_n = [[0 for i in range(len(V1))] for j in range(len(V2))]
            f_or = [[0 for i in range(len(V1))] for j in range(len(V2))]
            start = time.time()

        #We create the matrix; int64 since we run into troubles with int16 and big matrices
        matrix = np.zeros((dim_V2, dim_V1), dtype=np.int64)

        sign = 1
        for i in range(len(dim_bracket_V1)):
            Many =  self._lorigin( V1o[i], V2o)
            if self.doc:
                logger.debug('The bundle maps to {}'.format(Many))

            # y-position in the big matrix
            ymin = sum(dim_bracket_V1[:i])
            ymax = sum(dim_bracket_V1[:i+1])

            for j in range(len(Many)):
                if Many[j][0]:
                    # determine the minus sign for j
                    for k in range(len(V1o)):
                        if V1o[i][k] == j:
                            if k%2 == 0:
                                sign = 1
                                break
                            else:
                                sign = -1
                                break
                    x = self._single_map(V1[i], dim_bracket_V1[i], V2[Many[j][1]], dim_bracket_V2[Many[j][1]], j)
                    # fill the appropiate row in matrix
                    xmin = sum(dim_bracket_V2[:Many[j][1]])
                    xmax = sum(dim_bracket_V2[:Many[j][1]+1])
                    matrix[xmin:xmax,ymin:ymax] += sign*x
                    #if self.debug:
                    #    np.savetxt(str(i)+'to'+str(j)+'.csv', x, delimiter=',', fmt='%i')
                    #    print('xrange:', [xmin, xmax, xmax-xmin], 'yrange:',[ymin, ymax, ymax-ymin], 'mapshape:', x.shape)

                    if self.doc:
                        f_n[Many[j][1]][i] = [sign*abs(a-b) for a,b in zip(V1[i], V2[Many[j][1]])]
                        f_or[Many[j][1]][i] = [sign*j]                    

        if self.doc:
            mid = time.time()
            logger.info('Creation of the map took: {}.'.format(mid-start))
            if self.debug:
                if not os.path.exists(self.directory):
                    os.makedirs(self.directory)
                    #os.chdir(self.directory)
                    logger.debug('directory created at {}.'.format(self.cdirectory))
                tmp_dir = os.path.join(self.cdirectory, str(V1)+str(V2)+'.csv')
                np.savetxt(tmp_dir, matrix, delimiter=',', fmt='%i')
                logger.debug('Map has been saved at {}.'.format(tmp_dir))

        # Bottleneck for large matrices;
        # needs a lot of memory and takes the most time.
        if SpaSM:
            # increases the overhead, since the matrix has to be written to file
            # however, the faster rank computation should more than compensate for this.
            rank = self._spasm_rank(matrix)
        else:
            rank = np.linalg.matrix_rank(matrix)

        if self.doc:
            end = time.time()
            logger.info('The map in terms of polynomial degrees is given by \n {}.'.format(np.array(f_n)))
            logger.info('It has rank {}.'.format(rank))
            logger.info('Thus, dimension of kernel and image are {}.'.format([dim_V1-rank,rank]))
            logger.info('The rank calculation took {}.'.format(end-mid))
            logger.info('The total time needed for this map was {}.'.format(end-start))
            logger.debug('We had {} mapping via {} to {}.'.format(V1o, f_or, V2o))

        return [dim_V1-rank,rank]

    def _spasm_rank(self, matrix):
        r"""
        We use the SpaSM library to determine the rank.
        In order to use SpaSM, you need to compile the SpaSM code
        and link rank_hybrid to your $PATH

        This function creates a temporary file, which is then
        fed into SpaSM and subsequently deleted.

        Parameters
        ----------
        matrix : int_array[dimV2, dimV1]
            The map between two Leray entries.
        
        Returns
        -------
        int
            The rank of the map.
        """
        start = time.time()
        x_dim, y_dim = matrix.shape
        row, col = np.nonzero(matrix)
        
        tmp_file, filename = tempfile.mkstemp()
        # write file; tmp file needs utf-8 encoding.
        header = str(x_dim)+' '+str(y_dim)+' M\n'
        header = header.encode('utf-8')
        os.write(tmp_file, header)
        for r, c in zip(row, col):
            line = str(r+1)+' '+str(c+1)+' '+str(matrix[r,c])+'\n'
            line = line.encode('utf-8')
            os.write(tmp_file, line)
        close = '0 0 0'
        close = close.encode('utf-8')
        os.write(tmp_file, close)
        mid = time.time()
        # change directory; to spasm to run rank_hybrid
        old_dir = os.getcwd()
        full_filename = os.path.join(old_dir, filename)
        # run rank_hybrid
        output = subprocess.getoutput("cat {} | rank_hybrid".format(full_filename))
        # close the tmp_file
        os.close(tmp_file)
        end = time.time()
        logger.info('SpaSM info.\n File creation took {}, rank calculation {} and total time {}.'.format(mid-start, end-mid, end-start))
        # determine rank
        rank = ""
        # output looks like .... rank = XXX
        for n in reversed(output):
            if n != " ":
                rank += n
            else:
                #we break since all digits have been saved
                break
        # we need to flip since the string is in reversed order
        r = ""
        for n in reversed(rank):
            r += n
        # make to int
        r = int(r)
        return r

    def _makepoly(self, rep, dim):
        r"""Takes a bracket notation and creates a monomial basis.
        Schematically:
            (1,1) in the ambient space P1*P1 returns:
            - > [[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1]]
        Each entry denotes the exponent of the respective variable in the ambient space.

        Args:
            rep (list: int): A line bundle in bracket notation
            dim (int): the dimension of that line bundle
        
        Returns:
            (nested list: int): A basis of monomials
        """
        ambient = [0 for i in range(len(rep))]
        # create all possible ambient space polynomials
        # run over each ambient space degree
        for i in range(len(rep)):
            # create all polynomials of this degree   
            if abs(rep[i]) > 0:
                if rep[i] > 0:
                    ambient[i] = list(apoly(self.M[i][0]+1, rep[i]))
                else:
                    ambient[i] = list(apoly(self.M[i][0]+1, (-1)*rep[i]))
                    ambient[i] = [tuple(-x for x in y) for y in ambient[i]]
            else:
                #if zero there is only the zero combination
                ambient[i] = [tuple(0 for a in range(self.M[i][0]+1))]
        #create all combinations
        Base = [x for x in ambient[0]]
        for i in range(1,len(rep)):
            lenB = len(Base)
            Base = Base*len(ambient[i])
            for k in range(len(ambient[i])):
                for j in range(lenB):
                    #if we want to keep every ambient space in a seperate list
                    # we need to change here, might run into troubles with
                    # tuples here.
                    Base[k*lenB+j] = Base[k*lenB+j]+ambient[i][k]
        polynomial = np.array(Base, dtype=np.int8)
        return polynomial

    def line_index(self):
        r"""
        Determines the index of a general line bundle in terms of the charges m_i.
        Currently only implemented for three folds, where

        .. math::
            \begin{align}
            \text{ind}(L) = \sum_{q=0}^{n} (-1)^q h^q(X,L) = \frac{1}{6} d_{rst} m^r m^s m^t + \frac{1}{12} c_2^r m_r \; .
            \end{align}
        
        Returns
        -------
        euler: sympyexpr
            A polynomial in the line bundle charges

        See also
        --------
        line_co_euler: Determines the index of a specific line bundle.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.line_index()
        1.5*m0**2*m1 + 3.5*m0*m1**2 + 3.0*m0 + 0.333333333333333*m1**3 + 3.66666666666667*m1
        """
        L = sp.symbols('m0:'+str(self.len))
        if self.nfold == 3:
            euler = 0
            if self.triple.shape[0] == 0:
                self.triple = self.triple_intersection()
            for r in range(self.len):
                for s in range(self.len):
                    for t in range(self.len):
                        euler += self.triple[r][s][t]*(1/6*L[r]*L[s]*L[t]+1/12*L[r]*self.c2(s,t))
            return euler
        else:
            # TO DO implement this.
            logger.warning('line_index() is only implemented for 3-folds.')
            return 'not implemented yet'   

    def line_co_euler(self, L, Leray=False):
        r"""
        Determines the index of a line bundle L.
        
        Parameters
        ----------
        L : array[nProj]
            The line bundle L as a simple list.
        Leray : bool, optional
            If True, uses the Leray table to determine the index, by default False.
            For n=/=3 folds automatically falls back to the Leray table.
        
        Returns
        -------
        euler: float
            The index of L.

        See also
        --------
        line_index: Determines the index in terms of general charges.
        line_co: Determines the line bundle cohomology of L.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.line_co_euler([-4,3])
        -46.0
        """
        # using Chern classes
        if self.nfold == 3 and not Leray:
            line_tensor = 1/6*np.einsum('i,j,k -> ijk', L, L, L)
            chern_tensor = 1/12*np.einsum('i, jk -> ijk', L, self.c2_tensor)
            t = np.add(line_tensor, chern_tensor)
            return np.einsum('rst, rst', self.triple, t)

        # TO DO index of four folds with CHERN class
        # We use Leray

        # Build our Leray tableaux E_1[k][j]
        V = self._line_to_BBW(L)
        Table1, _ = self.Leray(V)        
        
        if self.doc:
            logger.info('Determine index via Lerray table.')
            t = Texttable()
            t.add_row(['j\\K']+[k for k in range(self.K, -1,-1)])
            for j in range(self.dimA+1):
                t.add_row([j]+[Table1[k][j] for k in range(self.K, -1,-1)])
            logger.info('\n'+t.draw())
        
        euler = 0
        for k in range(len(Table1)):
            for j in range(len(Table1[k])):
                if Table1[k][j] != 0:
                    for entry in Table1[k][j]:
                        euler += (-1)**(k+j)*self._brackets_dim(entry) 

        if self.doc:
            logger.info('The index is {}.'.format(euler))
        return euler

    def l_slope(self, line, dual=False, quick=True):
        r"""
        Determines the zero slope condition of a line bundle on a favourable CICY

        .. math::
            \begin{align}
            \mu (L) = c_1^i (L) d_{ijk} t^j t^k = 0 \; .
            \end{align}
        
        Parameters
        ----------
        line : array[nProj]
            The line bundle L.
        dual : bool, optional
            If true, uses dual coordinates k_i = d_{ijk} t^j t^k, by default False.
        quick : bool, optional
            If true skips the sympy expression and returns (bool, 0), by default True.
        
        Returns
        -------
        slope: bool
            True, if it can be satisfied somewhere in the Kähler cone.
        solution: sympyexpr
            The slope condition.

        See also
        --------
        line_slope: Returns the slope in terms of general charges.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.l_slope([-4,3], quick=False)
        (True, [9.0*t0**2 + 18.0*t0*t1 - 22.0*t1**2])
        """
        # define the Kähler moduli
        ts = sp.symbols('t0:'+str(len(line)))
        # find constraint
        constraint = 0

        if not self.fav:
            return False, constraint

        if not dual:

            mixed = False
            signs = np.einsum('ijk,i->jk', self.triple, line)
            signs = np.sign(signs+signs.T)
            if -1 in signs and 1 in signs:
                mixed = True
            else:
                mixed = False
            if quick:
                return mixed, 0

            # in terms of kähler moduli
            for i in range(self.len):
                for j in range(self.len):
                    for k in range(self.len):
                        factor = line[i]*self.drst(i,j,k,1)
                        constraint += factor*ts[j]*ts[k]

            if self.doc:
                logger.info('The slope stability constraint reads {}.'.format([constraint]))
            solution = [constraint]
            return mixed, solution
        
        # we use the dual coordinates
        for i in range(len(line)):
            constraint += line[i]*ts[i]
        slope = False
        if -1 in np.sign(line) and 1 in np.sign(line):
            slope = True

        #kaehlerc = [x > 0 for x in ts]
        if self.doc:
                logger.info('The slope stability constraint reads {}.'.format([constraint]))
        solution = [constraint]#+kaehlerc
        return slope, solution

    def line_slope(self):
        """
        Determines the slope of a general line bundle over a favourable CY.
        
        Returns
        -------
        constraint: sympyexpr
            Sympyexpression of the slope.

        See also
        --------
        l_slope: The zero slope condition of a line bundle L.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.line_slope()
        6.0*m0*t0*t1 + 7.0*m0*t1**2 + 3.0*m1*t0**2 + 14.0*m1*t0*t1 + 2.0*m1*t1**2
        """
        L = sp.symbols('m0:'+str(self.len))
        constraint = 0
        ts = sp.symbols('t0:'+str(self.len))

        if self.nfold == 3:
            for i in range(self.len):
                for j in range(self.len):
                    for k in range(self.len):
                        factor = L[i]*self.drst(i,j,k,1)
                        constraint += factor*ts[j]*ts[k]
            return constraint
        else:
            # TO DO. FIX THIS.
            logger.info('line_slope() is only implemented for 3-folds.')
            return 'not implemented'


    def line_co(self, L, short=True, SpaSM=False):
        """
        The main function of this CICY toolkit. 
        It determines the cohomology of a line bundle over the CY.
        Based on the Leray spectral sequence and Bott-Borel-Weil theorem. 
        By default makes use of the index and vanishing theorem
        to shorten computation time.
        
        Parameters
        ----------
        L : array[nProj]
            The line bundle L.
        short : bool, optional
            If False, calculates the rank of all maps and
             does not make use of simplifications, by default True.
             Don't use True for lb with 0-charges, as the code fails
             to pick up all relevant maps in those cases. The leray
             table implicitly assumes that there is only one non vanishing row.
             [TO DO FIX THIS]
        SpaSM : bool, optional
            If True, uses the SpaSM library to determine the rank,
             by default False.
        
        Returns
        -------
        hodge: array[nfold+1]
            hodge numbers of the line bundle L.
        hspace: nested list, optional
            If space, then also a nested list of maps is returned.

        See also
        --------
        line_co_euler: Returns the index of a line bundle

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.line_co([-4,3])
        [0,46,0,0]
        >>> #another example using SpaSM:
        >>> #assumes that rank_hybrid is in your $PATH
        >>> T = CICY([[1,2,0,0,0],[1,0,2,0,0],[1,0,0,2,0],[1,0,0,0,2],[3,1,1,1,1]])
        >>> T.line_co([3,-4,2,3,5], SpaSM=True)
        [496, 80, 0, 0]

        References
        ----------
        .. [1] CY - The Bestiary, T. Hubsch
            http://inspirehep.net/record/338506?ln=en

        .. [2] Heterotic and M-theory Compactifications for String Phenomenology, L. Anderson
            https://arxiv.org/abs/0808.3621

        .. [3] SpaSM: a Sparse direct Solver Modulo p.
            The SpaSM groub, http://github.com/cbouilla/spasm
        """
        # quick Kodaira
        if short:
            if np.array_equal(L, np.zeros(len(L))) and self.CY:
                return [1, 0, 0, 1]
            elif np.all(np.array(L) > 0):
                return np.round([self.line_co_euler(L), 0, 0, 0])
            elif np.all(np.array(L) < 0):
                return np.round([0, 0, 0, -1*self.line_co_euler(L)])

        # Build Leray tableaux E_1[k][j]
        start = time.time()
        V = self._line_to_BBW(L)
        Table1, origin = self.Leray(V)   

        logger.info('We determine the hodge numbers of {} over the CICY \n {}.'.format(L, self.M))
        logger.info('The first Leray instance takes the form:')
        t = Texttable()
        t.add_row(['j\\K']+[k for k in range(self.K, -1,-1)])
        for j in range(self.dimA+1):
            t.add_row([j]+[Table1[k][j] for k in range(self.K, -1,-1)])
        logger.info('\n'+t.draw())

        if self.debug:
            # change directory
            sdir = 'l'
            for a in L:
                sdir += str(a)
            self.cdirectory = os.path.join(self.directory, sdir)

        # variables for the image
        # there can be at most two rows in the Leray table with maps.
        images = [sp.symbols('f'+str(j)+r'\,(0:'+str(self.K+2)+')', integer=True) for j in range(self.dimA+1)]
        euler = 0
        E2 = sp.Matrix([[0 for j in range(self.dimA+1)] for k in range(self.K+1)])
        for j in range(self.dimA+1):
            for k in range(self.K+1):
                if Table1[k][j] != 0:
                    dimension = sum([self._brackets_dim(Table1[k][j][a]) for a in range(len(Table1[k][j]))]) 
                    euler += (-1)**(k+j)*dimension
                    E2[k,j] = dimension
                    # first the dim(kernel), which is dimension -image(k)
                    if k != 0 and Table1[k-1][j] != 0:
                        E2[k,j] -= images[j][k]
                    elif k == 0 and j < self.dimA and Table1[-1][j+1] != 0 and Table1[1][j] == 0:# 
                        # special case at the edge; assume generic map
                        E2[k,j] -= min(dimension, sum([self._brackets_dim(Table1[-1][j+1][a]) for a in range(len(Table1[-1][j+1]))]))
                    # then quotient out the image, -image(k+1)
                    if k < self.K and Table1[k+1][j] != 0:
                        E2[k,j] -= images[j][k+1]
                    elif k == self.K and Table1[0][j-1] != 0 and Table1[k-1][j] == 0:#
                        #special case at the edge; assume generic map
                        E2[k,j] -= min(dimension, sum([self._brackets_dim(Table1[0][j-1][a]) for a in range(len(Table1[0][j-1]))]))
        if 0 in L:
            #second order contribution for when there are zeros
            E2prime = E2.copy()
            contribution = np.ones(E2prime.shape).astype(np.bool)
            for j in range(self.dimA+1):
                for k in range(self.K+1):
                    if E2[k,j] != 0:
                        # upper tabular
                        jprime = 1
                        for p in range(k-2, -1,-1):
                            if j-jprime >= 0 and p >= 0 and Table1[p][j-jprime] != 0 and contribution[p, j-jprime]:
                                if type(E2[k,j]) is not int and type(E2[k,j]) is not sp.numbers.Integer:
                                    sol = {}
                                    for var in E2[k,j].free_symbols:
                                        if var == images[j][k]:
                                            sol[var] = self._rank_map(Table1[k][j], Table1[k-1][j], origin[k][j], origin[k-1][j], SpaSM)[1]
                                        elif var == images[j][k+1]:
                                            sol[var] = self._rank_map(Table1[k+1][j], Table1[k][j], origin[k+1][j], origin[k][j], SpaSM)[1]
                                    # now fill that in all
                                    E2 = E2.subs(sol)
                                    E2prime = E2prime.subs(sol)
                                dim1 = E2[k,j]
                                dim2 = sum([self._brackets_dim(Table1[p][j-jprime][a]) for a in range(len(Table1[p][j-jprime]))])
                                if dim1 > dim2:
                                    E2prime[k,j] -= E2[p,j-jprime]
                                    E2prime[p,j-jprime] = 0
                                contribution[p, j-jprime] = False
                            jprime += 1
                        # lower tabular
                        jprime = 1
                        for p in range(k+2, self.K+1):
                            if j+jprime < self.dimA+1 and Table1[p][j+jprime] != 0 and contribution[p, j+jprime]:
                                if type(E2[k,j]) is not int and type(E2[k,j]) is not sp.numbers.Integer:
                                    sol = {}
                                    for var in E2[k,j].free_symbols:
                                        if var == images[j][k]:
                                            sol[var] = self._rank_map(Table1[k][j], Table1[k-1][j], origin[k][j], origin[k-1][j], SpaSM)[1]
                                        elif var == images[j][k+1]:
                                            sol[var] = self._rank_map(Table1[k+1][j], Table1[k][j], origin[k+1][j], origin[k][j], SpaSM)[1]
                                    E2 = E2.subs(sol)
                                    E2prime = E2prime.subs(sol)
                                dim1 = E2[k,j]
                                dim2 = sum([self._brackets_dim(Table1[p][j+jprime][a]) for a in range(len(Table1[p][j+jprime]))])
                                if dim1 > dim2:
                                    E2prime[k,j] -= E2[p,j+jprime]
                                    E2prime[p,j+jprime] = 0
                                contribution[p, j+jprime] = False
                            jprime += 1
            E2 = E2prime

        #flatten images
        images = list(itertools.chain(*images))

        if self.doc:
            logger.info('The second Leray instance is \n {}.'.format(np.array(E2)))

        hodge = [0 for j in range(self.nfold+1)]
        done = True
        for q in range(self.nfold+1):
            for m in range(self.K+1):
                if m+q < self.dimA+1:
                    hodge[q] += E2[m,m+q]
            if type(hodge[q]) is not int:
                # then we have some maps
                done = False
        if done:
            if self.doc:
                end = time.time()
                logger.info('Thus we find h^*={}.'.format(hodge))
                logger.info('The calculation took {}.'.format(end-start))
            return hodge
        
        # now there is a theorem stating if L is slope stable,
        # then H^0 = H^3 = 0 by Serre.
        # only holds for favourable CY

        if self.fav and short:
            stable, _ = self.l_slope(L, dual=False)
        else:
            stable = False

        #Use euler + vanishing theorem to simplify
        if short:
            if self.nfold == 3:
                if stable:
                    solution = sp.solve([hodge[0], hodge[3], hodge[2]-hodge[1]-euler], images, dict=True)
                else:
                    solution = sp.solve(hodge[0]+hodge[2]-hodge[1]-hodge[3]-euler, images, dict=True)
            elif self.nfold == 4:
                if stable:
                    solution = sp.solve([hodge[0], hodge[4], hodge[2]-hodge[1]-euler-hodge[3]], images, dict=True)
                else:
                    solution = sp.solve(hodge[0]+hodge[2]+hodge[4]-hodge[1]-hodge[3]-euler, images, dict=True)
            elif self.nfold == 2:
                if stable:
                    solution = sp.solve([hodge[0], hodge[2], (-1)*hodge[1]-euler], images, dict=True)
                else:
                    solution = sp.solve(hodge[0]+hodge[2]-hodge[1]-euler, images, dict=True)

        if self.doc and short:
            logger.info('We find for the hodge numbers {}.'.format(hodge))
            logger.info('Index and slope stability impose the following constraints: {}'.format(solution))

        if short:
            # depending on scipy version solution is list or dictionary.
            if type(solution) is list:
                if len(solution) != 0:
                    for j in range(len(hodge)):
                        if type(hodge[j]) is not int:
                                hodge[j] = hodge[j].subs(solution[0])
            else:
                for j in range(len(hodge)):
                    if type(hodge[j]) is not int:
                            hodge[j] = hodge[j].subs(solution)

        if self.doc and short:
            logger.info('Thus we find h = {}'.format(hodge))

        # get all the maps we have to calculate
        maps = []
        for entry in hodge:
            if type(entry) is not int:
                maps += entry.free_symbols        
        maps = list(set(maps))

        # calculate all the maps
        # there could be a problem for line bundles with 0 charges when short=False
        maps_c = [0 for i in range(len(maps))]
        for i, m in enumerate(maps):
            name = m.name
            pos = name.find(',')
            j, k = int(name[1:pos]), int(name[pos+1:])
            maps_c[i] = self._rank_map(Table1[k][j], Table1[k-1][j], origin[k][j], origin[k-1][j], SpaSM)
            logger.info('The image {} has dimension {}.'.format(m, maps_c[i][1]))

        # substitute all values
        for i in range(len(maps)):
            for j in range(len(hodge)):
                if type(hodge[j]) is not int:
                    hodge[j] = hodge[j].subs(maps[i], maps_c[i][1])

        end = time.time()
        if self.doc:
            logger.info('Finally, we find h = {}.'.format(hodge))
            logger.info('The calculation took: {}.'.format(end-start))
        
        return hodge

#@staticmethod
def apoly( n, deg):
    if n == 1:
        yield (deg,)
    else:
        for i in range(deg + 1):
            for j in apoly(n - 1,deg - i):
                yield (i,) + j        
    
if __name__ == '__main__':
    
    print('done')
