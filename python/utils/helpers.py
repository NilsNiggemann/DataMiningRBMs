import numpy as np


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
    Jmat[0:total_bonds//d] = Js[0]
    Jmat[total_bonds//d:] = Js[1]
    if useH == True:
        hmat[:] = hs

    return [Jmat, hmat]
