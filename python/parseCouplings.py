import numpy as np
import os
import csv
from utils.helpers import makeJpairs

def parseCouplings(filename='couplings69.csv', M=4, N=4, d=2, BC='PBC', useH=True):
    '''
    reads couplings file, generates a list of Nsamples elements. 
    Each element of the list contains 3 objects: Jijalphabeta, h and bonds.
    Jijalphabeta is a list of 3x3 elements of size len(bonds)
    bonds is a list that contains the connecting elements of the graph
    h is a list of size number of sites = M * N
    BC can be both PBC and OBC, turning useH = False puts all magnetic fields to zero
    '''
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return np.array([makeJpairs(couplings, M, N, d, BC, useH) for couplings in reader], dtype=object).T

def main():
    Jijalphabeta, h, bonds = parseCouplings()
    print(bonds)

if __name__ == '__main__':
    main()
