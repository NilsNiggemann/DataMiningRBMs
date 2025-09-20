# Documentation for generating coupling constants

## Description of lattice 
description of lattice : square lattice of size M X N in d = 2 dimensions
 total number of bonds 
       PBC : M X N X d 
       OBC : (M-1) X (N-1) X d 

Bonds in the X direction numbered 0 ---> (M X N) - 1
Bonds in the Y direction numbered M X N ---> M X N X 2 - 1 

The sites are numbered from 0 ---> (M X N)
The bonds are numbered so that 0 --> total_bonds//2 are in the X direction and the rest are in the Y direction of the square lattice.

For example, for a bond numbered i, if (i//total_bonds) == 0, it represents and X - direction bond that connects
site i to site (i+1) % N in the case of PBC. In coordinates, this would be 
[i//N , i%N] --> [i//N, (i+1)%N]

## genCouplings.py 
Uses a fixed SEED (=69 by default) for ease of data reproducibility. For a 2d
square lattice, it generates 18 + 3 = 21 random numbers from a normal
distribution with zero mean and unit standard deviation. 
For each instance of 21 numbers, a line is written to a csv file.
A backup is taken in case a mistake occurs and the file gets overwritten
Only needs to be run once after setting the number of samples (~1000) in the main()

## utils.helpers.makeJmat
not requried for the end user

## parseCouplings.py Uses the generated couplings from the stored csv file and
creates a two lists of size nsamples each 
1) Jlist : each Jlist[i] contains in
the first half total_bonds//2 Jx matrices and total_bonds//2 Jy matrices in the
second half, that are themselves of size 3x3, representing the different
J_{\alpha\beta} 
2) hlist : each hlist[i] contains total_sites number of size 3
vectors containing {hx,hy,hz}. 

There is an option to turn off magnetic fields, in which case the random
numbers corresponding to h are just ignored, but still generated, so that they
can be reproduced with the same SEED if required. 




