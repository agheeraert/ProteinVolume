from os.path import join as jn
import numpy as np
import pickle as pkl
import argparse
import warnings
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Smooth the average plots by plotting the moving average')
parser.add_argument('-f',  type=str, nargs='+',
                    help='List of input files')
parser.add_argument('-n', type=int, default=10,
		    help='Smoothing window size')

args = parser.parse_args()

def moving_average(table, n) :
    convo = np.convolve(table, np.ones((n+1,))/(n+1), mode='valid')
    nanarray = np.empty(n)
    nanarray[:] = np.nan
    return np.concatenate([convo, nanarray], axis=-1)

def plot_apo_holo(table, n_trajs, n_frames, output):
    f = plt.figure()
    table_apo = table[n_trajs//2*n_frames:]
    table_holo = table[:n_trajs//2*n_frames]
    previous = 0
    for i in range(n_trajs//2):
        if n_trajs//4 >= 1:
            plt.subplot(2, n_trajs//4, i+1)
        plt.title("Simulation "+str(i+1))
        plt.plot(table_apo[previous:previous+n_frames], c='b', label="apo")
        plt.plot(table_holo[previous:previous+n_frames], c='r', label="PRFAR")
        previous += n_frames
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

n_frames = 1000
n_sims = 8
for f in args.f:
    mean = np.array(pkl.load(open(f, "rb")))
    moving_mean = []
    for sim in range(n_sims):
        moving_mean.append(moving_average(mean[sim*n_frames:(sim+1)*n_frames], args.n))
    moving_mean = np.concatenate(moving_mean, axis=-1)
    if f[-2:] == '.p':
        plot_apo_holo(moving_mean, n_sims, n_frames, f.replace('.p', '_smoothed'+str(args.n)+'.png'))
    else:
        plot_apo_holo(moving_mean, n_sims, n_frames, f+'_smoothed'+str(args.n)+'.png')
