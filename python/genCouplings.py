import numpy as np
import csv
import os


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

def main():
    nsamples = 10
    makeSamples(nsamples)
    
    

if __name__ == '__main__':
    main()









