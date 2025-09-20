import numpy as np

N,M = (2,2)
d = 2 
SEED = 69
rng = np.random.default_rng(SEED)
Jmat = rng.normal(loc=0.0, scale=1.0, size=(d,3,3))


#description of lattice : square lattice of size M X N in d = 2 dimensions
# total number of bonds 
#       PBC : M X N X d 
#       OBC : (M-1) X (N-1) X d 

#Bonds in the X direction numbered 0 ---> (M X N) - 1
#Bonds in the Y direction numbered M X N ---> M X N X 2 - 1 


J = rng.normal(0,1)
print(J)
