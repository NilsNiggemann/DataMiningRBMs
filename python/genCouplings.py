import numpy as np
import csv
import sys
import os

#description of lattice : square lattice of size M X N in d = 2 dimensions
# total number of bonds 
#       PBC : M X N X d 
#       OBC : (M-1) X (N-1) X d 

#Bonds in the X direction numbered 0 ---> (M X N) - 1
#Bonds in the Y direction numbered M X N ---> M X N X 2 - 1 


SEED = 69 # Random numbers generated are identical in each run
rng = np.random.default_rng(SEED)


def retCouplingsSqLat(d=2):
    Js = rng.normal(loc=0.0, scale=1.0, size=(d*3*3 + 3)) #Js[0] = Jx, Js[1] = Jy
    # hs = rng.normal(loc=0.0, scale=1.0, size=(3,)) # hs[0] = hx, hs[1] = hy, hs[2] = hz
    return Js

def makeSamples(nsamples=1, savefile='couplings'+str(SEED)+'.csv'):
    if os.path.exists(savefile):
        os.rename(savefile, savefile+'.bak')

    with open(savefile,'w') as f:
        writer = csv.writer(f)
        for i in range(nsamples):
            couplings = retCouplingsSqLat()
            writer.writerow(couplings)


def makeJmat(couplings, M=4,N=4,d=2,BC='PBC',useH=True):
    ''' acts on one line of couplings.csv and converts it to the required tensor '''

    if BC == 'PBC':
        total_bonds = M * N * d
        total_sites = M * N
        shape = (M,N,3,3)
    elif BC == 'OBC':
        total_bonds = (M-1) * (N-1) * d
        total_sites = N * N
        shape = (M-1,N-1,3,3)
    else:
        raise(Exception("invalid choice of boundary condition: choices PBC, OBC "))
    

    Js = np.array(couplings[:-3])
    assert len(Js)==3*3*d, 'Not the correct number of Js, splitting of couplings improperly done!'
    Js = Js.reshape((d,3,3))
    hs = np.array(couplings[-3:])
    assert len(hs)==3, 'Not the correct number of hs, splitting of couplings improperly done!'

    Jmat = np.zeros((total_bonds,3,3))
    hmat = np.zeros((total_sites,3))
    Jmat[0:total_bonds//2] = Js[0]
    Jmat[total_bonds//2:] = Js[1]
    if useH == True:
        hmat[:] = hs

    # print(Js.reshape((2,3,3)))
    return (Jmat, hmat)

def parse_couplings(filename='couplings69.csv', M=4, N=4, d=2, BC='PBC', useH=True):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for couplings in reader:
            coupmat = makeJmat(couplings, M, N, d, BC, useH)

    



def main():
    makeSamples(1)
    with open('couplings69.csv', 'r') as f:
        reader = csv.reader(f)
        for couplings in reader:
            makeJmat(couplings)
    
    

if __name__ == '__main__':
    main()









