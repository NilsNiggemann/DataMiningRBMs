import numpy as np
import csv


def makeJmat(couplings, M=4, N=4, d=2, BC='PBC', useH=True):
    ''' acts on one line of couplings.csv and converts it to the required tensor '''

    if BC == 'PBC':
        total_bonds = M * N * d
        total_sites = M * N
        shape = (M, N, 3, 3)
    elif BC == 'OBC':
        total_bonds = (M-1) * (N-1) * d
        total_sites = N * N
        shape = (M-1, N-1, 3, 3)
    else:
        raise (Exception("invalid choice of boundary condition: choices PBC, OBC "))

    Js = np.array(couplings[:-3])
    assert len(
        Js) == 3*3*d, 'Not the correct number of Js, splitting of couplings improperly done!'
    Js = Js.reshape((d, 3, 3))
    hs = np.array(couplings[-3:])
    assert len(
        hs) == 3, 'Not the correct number of hs, splitting of couplings improperly done!'

    Jmat = np.zeros((total_bonds, 3, 3))
    hmat = np.zeros((total_sites, 3))
    Jmat[0:total_bonds//d] = Js[0]
    Jmat[total_bonds//d:] = Js[1]
    if useH == True:
        hmat[:] = hs

    return [Jmat, hmat]

def makeJpairs(couplings, M=4, N=4, d=2, BC='PBC', useH=True):
    ''' for the format needed by construct_hamiltonian_bonds'''
    total_sites = M * N
    if BC == 'PBC':
        total_bonds = M * N * d
        shape = (M, N, 3, 3)
    elif BC == 'OBC':
        total_bonds = (M-1) * (N-1) * d
        shape = (M-1, N-1, 3, 3)
    else:
        raise (Exception("invalid choice of boundary condition: choices PBC, OBC "))
    Js = np.array(couplings[:-3])
    assert len(
        Js) == 3*3*d, 'Not the correct number of Js, splitting of couplings improperly done!'
    Js = Js.reshape((d, 3, 3))  # J[0] = Jx, J[1] = Jy
    hs = np.array(couplings[-3:])
    assert len(
        hs) == 3, 'Not the correct number of hs, splitting of couplings improperly done!'

    ### OBC ###
    ybonds = [(i, i+1) for i in range(total_sites) if (i+1) % M != 0]
    xbonds = [(i, i+M) for i in range(total_sites) if (i+M) < total_sites]
    if BC == 'PBC':
        ### PBC contains additional bonds ###
        xper = [(i, i % N)
                for i in range(total_sites - M, total_sites)]  # note order
        yper = [(i, i+1-M) for i in range(M-1, total_sites, N)]  # note order
        # bonds = [pair for pair in [xbonds + xper + ybonds + yper]]
        bonds = xbonds + xper + ybonds + yper
    else:
        # bonds = [pair for pair in [xbonds + ybonds]]
        bonds = xbonds + ybonds

    assert len(bonds) == total_bonds, print(f'ERROR IN BONDS CREATION!!!!! {len(bonds), total_bonds}')
    Jmat = np.zeros((total_bonds, 3, 3))
    hmat = np.zeros((total_sites, 3))
    Jmat[0:total_bonds//d] = Js[0]
    Jmat[total_bonds//d:] = Js[1]
    if useH == True:
        hmat[:] = hs
    return (Jmat, hmat, bonds)


if __name__ == '__main__':
    with open('testcouplings.csv', 'r') as f:
        couplings = [np.float64(cval) for cval in f.readline().split(',')]
    result = makeJpairs(couplings)
    print(result)
