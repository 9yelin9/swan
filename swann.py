#!/home/9yelin9/.local/bin/python3

swann_path = '/home/9yelin9/swann'
band_path = 'swann_band'
lat_columns = ['i', 'j', 'k', 'p', 'q', 't_real', 't_imag']
lat_dtypes = {'i':'i', 'j':'i', 'k':'i', 'p':'i', 'q':'i', 't_real':'d', 't_imag':'d'}

import os
num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
print('num_threads =', num_threads, end='\n\n')

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--atom', type=str, required=True)
parser.add_argument('-wl', '--wann2lat',  type=int, nargs='+', help='Wann2Lat: <n(0 or int)> <show_tR=0/1>')
parser.add_argument('-gb', '--genband',  type=str, nargs='+', help='GenBand: <path_lat> <Nk>')
parser.add_argument('-sb', '--showband', type=str, nargs='+', help='ShowBand: <path_band(sep=:)>')
args = parser.parse_args()                                                                     

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})

import re
import ctypes
import cycler
import numpy as np
import pandas as pd

Dim = 3
print('Dim =', Dim, end='\n\n')

def ReSub(pattern, string):
    return float(re.sub(pattern, '', re.search('%s[-]?\d+[.]\d+' % pattern, string).group()))

def ReSubInt(pattern, string):
    return int(re.sub(pattern, '', re.search('%s[-]?\d+' % pattern, string).group()))

def ReadInput(atom):
    path, pos0, A = [], [], []

    with open('swann_input.txt', 'r') as f:
        read_line = 0
        for line in f:
            if re.search('end kpoint_path', line): break
            elif read_line: path.append(line.split())
            elif re.search('begin kpoint_path', line): read_line = 1

        read_line = 0
        for line in f:
            if re.search('end unit_cell_cart', line): break
            elif read_line: A.append(line.split())
            elif re.search('begin unit_cell_cart', line): read_line = 1
        A = np.array(A).astype('d')
        A /= np.max(A)

        read_line = 0
        for line in f:
            if re.search('end atoms_frac', line): break
            elif read_line:
                if re.search(atom, line): pos0.append(line.split()[1:])
            elif re.search('begin atoms_frac', line): read_line = 1
        pos0 = np.array(pos0).astype('d')

    pos = np.zeros(pos0.shape)
    for i, p in enumerate(pos0):
        for j in range(Dim): pos[i] = np.add(pos[i], p[j]*A[j])

    return path, pos, A 

path, pos, A = ReadInput(args.atom)
print('path =', *path, sep='\n')
print('pos =', *pos, sep='\n')
print('A =', *A, sep='\n', end='\n\n')

def Wann2Lat(n, show_tR=0):
    pat_site = '[-]?\d+\s+'
    pat_obt  = '[-]?\d+\s+'
    pat_t    = '[-]?\d+[.]\d+\s+'
    pat = Dim * pat_site + 2 * pat_obt + 2 * pat_t

    with open('wannier90_hr.dat', 'r') as f:
        for line in f: 
            if re.search(pat, line): break
        df = pd.read_csv(f, sep='\s+', names=lat_columns).astype(lat_dtypes)

    if n:
        #df['p_pos'], df['q_pos'] = (df['p']-1) // Nc, (df['q']-1) // Nc
        R = []
        for _, d in df.iterrows():
            #r = np.linalg.norm(d['i']*A[0] + d['j']*A[1] + d['k']*A[2] + pos[int(d['q_pos'])] - pos[int(d['p_pos'])])
            r = np.linalg.norm(d['i']*A[0] + d['j']*A[1] + d['k']*A[2])
            R.append(r)
        df['R'] = R

        R = np.unique(np.round(R, decimals=6))[:n+1]
        df = df[df['R'] < R[n]]

    if show_tR:
        df['t'] = np.sqrt(df['t_real']**2 + df['t_imag']**2)

        R1 = R[1]
        t1_max = np.max(df[np.abs(df['R']-R1) < 1e-6]['t'])

        print('%6s%16s%16s' % ('n', 'R/R1', 't_max/t1_max'))
        for i in range(1, len(R)):
            t_max = np.max(df[np.abs(df['R']-R[i]) < 1e-6]['t'])
            print('%6d%16.6f%16.6f' % (i, R[i]/R1, t_max/t1_max))
        print()

    fn = 'lattice_n%d.txt' % n
    np.savetxt(fn, df[lat_columns], fmt=['%5d', '%5d', '%5d', '%5d', '%5d', '%12.6f', '%12.6f'])
    print('File saved at %s' % fn)

def GenK(Nk):
    path_label, path_point = [], []
    for p in path:
        path_label.append([p[0],       p[Dim+1]])
        path_point.append([p[1:Dim+1], p[Dim+2:]])
    path_label, path_point = np.array(path_label), np.array(path_point, dtype='d')

    k_label = np.append(path_label[:, 0], path_label[-1, 1])

    dist = []
    for pi, pf in path_point:
        ki, kf = np.zeros(Dim), np.zeros(Dim)
        for i in range(Dim):
            ki = np.add(ki, pi[i]*A[i])
            kf = np.add(kf, pf[i]*A[i])
        dist.append(np.linalg.norm(ki - kf))
    dist = [int((Nk-1)*d / sum(dist)) for d in dist]
    dist[-1] += (Nk-1) - sum(dist)
    k_point = [0] + list(np.cumsum(dist))

    k = []
    for (pi, pf), d in zip((2*np.pi)*path_point, dist):
        Nd = d
        if d == dist[-1]: Nd = d + 1
        for di in range(Nd): k.append([pi[i] + (pf[i] - pi[i]) * di/d for i in range(Dim)])

    return k, k_label, k_point

def GenBand(path_lat, Nk):
    Nk = int(Nk)
    print('path_lat =', path_lat, '\nNk =', Nk, end='\n\n')

    df = pd.read_csv(path_lat, sep='\s+', names=lat_columns).astype(lat_dtypes)
    Nb = int(np.max(df['p']))
    k, k_label, k_point = GenK(Nk)

    site_c = np.ctypeslib.as_ctypes(np.ravel(df[['i', 'j', 'k']].astype(dtype='i')))
    obt_c  = np.ctypeslib.as_ctypes(np.ravel(df[['p', 'q']].astype(dtype='i')))
    t_c    = np.ctypeslib.as_ctypes(np.ravel(df[['t_real', 't_imag']]))
    k_c    = np.ctypeslib.as_ctypes(np.ravel(k))
    band_c = np.ctypeslib.as_ctypes(np.ravel(np.zeros((Nk, Nb))))

    swann_c = ctypes.cdll.LoadLibrary('%s/libswann.so' % swann_path)
    swann_c.GenBand(num_threads, Dim, Nb, Nk, len(df), k_c, site_c, obt_c, t_c, band_c)
    band = np.reshape(np.ctypeslib.as_array(band_c), (Nk, Nb))

    os.makedirs(band_path, exist_ok=True)
    fn = '%s/band_%s_Nk%d.dat' % (band_path, re.sub('.txt', '', re.sub('lattice_', '', path_lat)), Nk)
    np.savetxt(fn, band)
    print('File saved at %s' % fn)

def ShowBand(path_band):
    path_band = path_band.split(':')
    print('path_band =', *path_band, sep='\n', end='\n\n')

    Nk = ReSubInt('Nk', path_band[0])
    k, k_label, k_point = GenK(Nk)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    color = plt.cm.tab10(range(10))
    ls = ['-', '--', '-.', ':'] * 2

    for i, p in enumerate(path_band):
        band, n = np.genfromtxt(p), ReSubInt('n', p)
        label = 'n=%d' % n 

        if i:
            label += ', $|E_{%d}-E_{%d}|_\mathrm{max}$=%.4f' % (n0, n, np.max(np.abs(band0 - band)))
            band_max, band_scale = np.max(band), np.max(band)-np.min(band)
            dband = band0_max - band_max; #band += dband; band *= band_scale/band0_scale
        else:
            band0, band0_max, band0_scale, n0 = band, np.max(band), np.max(band)-np.min(band), n 
            Nb = band0.shape[1]

        ax.plot(range(Nk), band[:, 0], color=color[i], ls=ls[i], label=label)
        for j in range(1, Nb): ax.plot(range(Nk), band[:, j], color=color[i], ls=ls[i])
    ax.grid(True, axis='x')
    ax.legend(fontsize='xx-small')
    ax.set_ylim([np.min(band0)-0.5, np.max(band0)+0.5])
    ax.set_xticks(k_point, labels=k_label)
    ax.set_ylabel(r'$E$')

    fn = '%s/band_' % band_path + '_'.join(['n%d' % ReSubInt('n', p) for p in path_band]) + '_Nk%d.png' % Nk
    fig.savefig(fn)
    print('Figure saved at %s' % fn)
    plt.show()

if args.wann2lat: Wann2Lat(*args.wann2lat)
elif args.genband: GenBand(*args.genband)
elif args.showband: ShowBand(*args.showband)
else: parser.print_help()
