The content of the folder SemiGlobal
-------------------------------------
The folder contains a Python implementation of the semi-global propagator, with examples
and theory.

The folder SemiGlobal contains the Python package.
The main function is SemiGlobal in the main module SGfuns.py.

The modules Arnoldi.py, Chebyshev.py and NewtonIpln.py contain functions which are used
by the code.

The module FourierGrid.py is required for the examples in the subfolder examples.


The content of the subfolder examples
--------------------------------------
The folder contains several examples for the application of the SemiGlobal function,
in several script files.

test_harmonic.py tests the propagator for a forced harmonic oscillator problem.

testBECsg.py is similar to test_harmonic.py, but with the addition of a nonlinear BEC trap 
potential.

test_source_term.py is similar to test_harmonic.py, but with the addition of an arbitrary 
source term.

decay_curves1DHydrogen.py reproduces the results of the paper attached in the folder doc
(originally obtained by a MATLAB implementation of the algorithm).

H1Dpython.mat is a MATLAB data file; it contains some data which is used by the script
decay_curves1DHydrogen.py.


The content of the subfolder doc
---------------------------------
The folder contains the mathematical and numerical background of the algorithm.
SemiGlobal.pdf is the author-made version of the following paper:
Ido Schaefer, Hillel Tal-Ezer and Ronnie Kosloff,
"Semi-global approach for propagation of the time-dependent Schr\"odinger 
equation for time-dependent and nonlinear problems", JCP 2017
This paper gives provides a broad exposition of the theory of the algorithm, and practical
details on its numerical implementation.
A pseudo code appears in Sec. 3.2 (where a few "corrections" to the pseudo code for 
the sake of numerical stability appear in Sec. 3.3.1).

Some additional theoretical details on the computation of the f_m(z,t) functions appear in
f_error.pdf.
