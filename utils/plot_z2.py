'''
'''

import argparse
import numpy as np
import sys, os
import pickle
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='+', help='Input')
    parser.add_argument('-o', help='Output')
    parser.add_argument('--sample1', type=int, help='Plot z value for N points')
    parser.add_argument('--sample2', type=int, help='Plot median z after chunking into N chunks')
    parser.add_argument('--out-s', help='Save sampled z values')
    parser.add_argument('--equiv', action='store_true')
    return parser

def main(args):
    for f in args.input:
        print(f)
        fi = open(f,'rb')
        x = pickle.load(fi)
        plt.scatter(x[:,0], x[:,1], c=np.arange(len(x[:,0])), label=f, alpha=.1, s=2, cmap='hsv')
        if args.equiv:
            plt.figure()
            y = pickle.load(fi)
            plt.scatter(y[:,0], y[:,1], c=np.arange(len(x[:,0])), label=f, alpha=.1, s=2, cmap='hsv')
            plt.figure()
            z = x-y
            plt.scatter(z[:,0], z[:,1], c=np.arange(len(x[:,0])), label=f, alpha=.1, s=2, cmap='hsv')
    if args.sample1:
        d = len(x) // args.sample1
        xd = x[::d]
        print(len(xd))
        print(xd)
        plt.scatter(xd[:,0],xd[:,1],c=np.arange(len(xd)),cmap='hsv')
    if args.sample2:
        xsplit = np.array_split(x,args.sample2)
        xd = np.array([np.median(xs,axis=0) for xs in xsplit])
        print(len(xd))
        print(xd)
        plt.scatter(xd[:,0],xd[:,1],c='k')#np.arange(len(xd)),cmap='hsv')
    if args.out_s:
        np.savetxt(args.out_s, xd)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend(loc='best')
    if args.o: 
        plt.savefig(args.o)
    else:
        plt.show()

if __name__ == '__main__':
    main(parse_args().parse_args())