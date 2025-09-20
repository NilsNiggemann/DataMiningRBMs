import numpy as np
import os
import csv
from utils.helpers import makeJmat

def parseCouplings(filename='couplings69.csv', M=4, N=4, d=2, BC='PBC', useH=True):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return np.array([makeJmat(couplings, M, N, d, BC, useH) for couplings in reader], dtype=object).T

def main():
    Jlist, hlist = parseCouplings()

if __name__ == '__main__':
    main()
