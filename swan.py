#!/home/9yelin9/.local/bin/python3

swan_path = '/home/9yelin9/swan'
band_path = 'band'

import os
num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
print('num_threads =', num_threads, end='\n\n')

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--atom', type=str, required=True)
parser.add_argument('-st', '--showtR',   type=str, nargs='+', help='ShowtR: <n(0 or int)>')
parser.add_argument('-gb', '--genband',  type=str, nargs='+', help='GenBand: <n(0 or int)> <Nk>')
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
    Nb, pos0, A = -1, [], []

    with open('wannier90.win', 'r') as f:
        for line in f:
            if re.search('num_wann', line): Nb = int(line.split()[-1]); break

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

    return Nb, pos, A 

Nb, pos, A = ReadInput(args.atom)
Ni = len(pos)
Nc = Nb // Ni

print('Nb =', Nb, '\nNi =', Ni, '\nNc =', Nc)
print('pos =\n', pos, '\nA =\n', A, end='\n\n')

def ReadLattice(n):
    pat_site = '[-]?\d+\s+'
    pat_obt  = '[-]?\d+\s+'
    pat_t    = '[-]?\d+[.]\d+\s+'
    pat = Dim * pat_site + 2 * pat_obt + 2 * pat_t

    with open('wannier90_hr.dat', 'r') as f:
        for line in f: 
            if re.search(pat, line): break
        df = pd.read_csv(f, sep='\s+', names=['i', 'j', 'k', 'p', 'q', 't_real', 't_imag'])

    df['t'] = np.sqrt(df['t_real']**2 + df['t_imag']**2)
    df['p_pos'], df['q_pos'] = (df['p']-1) // Nc, (df['q']-1) // Nc

    R = []
    for _, d in df.iterrows():
        r = np.linalg.norm(d['i']*A[0] + d['j']*A[1] + d['k']*A[2] + pos[int(d['q_pos'])] - pos[int(d['p_pos'])])
        R.append(r)
    df['R'] = R

    R = np.unique(np.round(R, decimals=6))
    if n:
        R = R[:n+1]
        df = df[df['R'] < R[n]]

    return df, R

def ReadK(Nk):
    path_label, path_point = [], []

    with open('wannier90.win', 'r') as f:
        read_line = 0
        for line in f:
            if re.search('end kpoint_path', line): break
            elif read_line:
                lines = line.split()
                path_label.append([lines[0],       lines[Dim+1]])
                path_point.append([lines[1:Dim+1], lines[Dim+2:]])
            elif re.search('begin kpoint_path', line): read_line = 1
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

def ShowtR(n):
    n = int(n)
    print('n =', n, end='\n\n')

    df, R = ReadLattice(n)
    R1 = R[1]
    t1_max = np.max(df[np.abs(df['R']-R1) < 1e-6]['t'])

    print('%6s%16s%16s' % ('n', 'R/R1', 't_max/t1_max'))
    for i in range(1, len(R)):
        t_max = np.max(df[np.abs(df['R']-R[i]) < 1e-6]['t'])
        print('%6d%16.6f%16.6f' % (i, R[i]/R1, t_max/t1_max)) 

def GenBand(n, Nk):
    n, Nk = int(n), int(Nk)
    print('n =', n, '\nNk =', Nk, end='\n\n')

    df, R = ReadLattice(n)
    k, k_label, k_point = ReadK(Nk)

    site_c = np.ctypeslib.as_ctypes(np.ravel(df[['i', 'j', 'k']].astype(dtype='i')))
    obt_c  = np.ctypeslib.as_ctypes(np.ravel(df[['p', 'q']].astype(dtype='i')))
    t_c    = np.ctypeslib.as_ctypes(np.ravel(df[['t_real', 't_imag']]))
    k_c    = np.ctypeslib.as_ctypes(np.ravel(k))
    band_c = np.ctypeslib.as_ctypes(np.ravel(np.zeros((Nk, Nb))))

    swan_c = ctypes.cdll.LoadLibrary('%s/libswan.so' % swan_path)
    swan_c.GenBand(num_threads, Dim, Nb, Nk, len(df), k_c, site_c, obt_c, t_c, band_c)
    band = np.reshape(np.ctypeslib.as_array(band_c), (Nk, Nb))

    os.makedirs(band_path, exist_ok=True)
    fn = '%s/band_n%d_Nk%d.dat' % (band_path, n, Nk)
    np.savetxt(fn, band)
    print('File saved at %s' % fn)

def ShowBand(path_band):
    path_band = path_band.split(':')
    Nk = ReSubInt('Nk', path_band[0])
    k, k_label, k_point = ReadK(Nk)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for p, c in zip(path_band, plt.cm.tab10(np.linspace(0, 1, len(path_band)))):
        band = np.genfromtxt(p)
        label = 'n=%d' % ReSubInt('n', p)
        ax.plot(range(Nk), band[:, 0], color=c, label=label)
        for i in range(1, Nb): ax.plot(range(Nk), band[:, i], color=c)
    ax.grid(True, axis='x')
    ax.legend(fontsize='xx-small')
    ax.set_xticks(k_point, labels=k_label)
    ax.set_ylabel(r'$E-E_{F}$')

    fn = '%s/band_' % band_path + '_'.join(['n%d' % ReSubInt('n', p) for p in path_band]) + '_Nk%d.png' % Nk
    fig.savefig(fn)
    print('Figure saved at %s' % fn)
    plt.show()

if args.showtR: ShowtR(*args.showtR)
elif args.genband: GenBand(*args.genband)
elif args.showband: ShowBand(*args.showband)
else: parser.print_help()
