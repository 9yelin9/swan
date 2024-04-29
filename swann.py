#!/home/9yelin9/.local/bin/python3

swann_path = '/home/9yelin9/swann'
band_path = 'band_swann'
lat_cols = ['i', 'j', 'k', 'p', 'q', 't_real', 't_imag', 'norm']
lat_dtypes = {'i':'i', 'j':'i', 'k':'i', 'p':'i', 'q':'i', 't_real':'d', 't_imag':'d', 'norm':'d'}

import os
num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
print('num_threads =', num_threads, end='\n\n')

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--atom', type=str, required=True)
parser.add_argument('-wl', '--wann2lat', type=int, nargs='+', help='Wann2Lat: <n(0/int)> [show_sym=0/1] [show_tR=0/1]')
parser.add_argument('-gb', '--genband',  type=str, nargs='+', help='GenBand: <path_lat> <Nk> [show_band=0/1]')
parser.add_argument('-sb', '--showband', type=str, nargs='+', help='ShowBand: <path_band(sep=:)>')
args = parser.parse_args()                                                                     

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})

import re
import ctypes
import cycler
import numpy as np
import pandas as pd

def ReSub(pattern, string):
    return float(re.sub(pattern, '', re.search('%s[-]?\d+[.]\d+' % pattern, string).group()))

def ReSubInt(pattern, string):
    return int(re.sub(pattern, '', re.search('%s[-]?\d+' % pattern, string).group()))

def ReadWin(atom):
    Dim, Nb, Nc, Ni = Dim, 0, 0, 0
    path, pos0, basis, A = [], [], [], []

    with open('wannier90.win', 'r') as f:
        read_line = 0
        for line in f:
            if re.search('end kpoint_path', line): break
            elif read_line: path.append(line.split())
            elif re.search('begin kpoint_path', line): read_line = 1

        read_line = 0
        for line in f:
            if re.search('end projections', line): break
            elif read_line:
                pos0.append(re.sub('f=|:.+', '', line).split(','))
                basis.append(re.sub('.+:|\n', '', line))
            elif re.search('begin projections', line): read_line = 1
        Nb, Nc = len(basis), len(set(basis)); Ni = Nb // Nc
        pos0 = np.round(np.array(pos0, dtype='d'), decimals=6)[::Nc]

        read_line = 0
        for line in f:
            if re.search('end unit_cell_cart', line): break
            elif read_line: A.append(line.split())
            elif re.search('begin unit_cell_cart', line): read_line = 1
        A = np.array(A, dtype='d')
        A /= A[A > 0][0]
        Dim = len(A)
        pos = np.array([np.sum([p[i]*A[i] for i in range(Dim)], axis=0) for p in pos0])

    return Dim, Nb, Nc, Ni, path, pos, A

Dim, Nb, Nc, Ni, path, pos, A = ReadWin(args.atom)
print('Dim =', Dim, 'Nb =', Nb, 'Nc =', Nc, 'Ni =', Ni)
print('path =', *path, sep='\n')
print('pos =', *pos, sep='\n')
print('A =', *A, sep='\n', end='\n\n')

def Wann2Lat(n, show_sym=0, show_tR=0):
    pat_site = '[-]?\d+\s+'
    pat_obt  = '[-]?\d+\s+'
    pat_t    = '[-]?\d+[.]\d+\s+'
    pat = Dim * pat_site + 2 * pat_obt + 2 * pat_t

    with open('wannier90_hr.dat', 'r') as f:
        for line in f: 
            if re.search(pat, line): break
        df = pd.read_csv(f, sep='\s+', names=lat_cols[:-1]).astype(dict(list(lat_dtypes.items())[:-1]))

    r_arr = np.array([np.sum([d[i]*A[i] for i in range(Dim)], axis=0) + pos[(d.q-1)//Nc] - pos[(d.p-1)//Nc] for d in df.itertuples(index=False)])
    r_norm = np.round(np.linalg.norm(r_arr, axis=1), decimals=6)
    df['norm'] = r_norm
    #df['p'], df['q'] = (df['p']-1) // Nc, (df['q']-1) // Nc
    df.sort_values(by=['norm', 'i', 'j', 'k', 'p', 'q'], inplace=True)

    if n:
        r_norm_uq = np.unique(r_norm)[:n+2]
        r_arr = r_arr[r_norm < r_norm_uq[n+1]]; #r_arr_uq = np.unique(r_arr, axis=0)
        df = df[df['norm'] < r_norm_uq[n+1]]
        r_norm = r_norm_uq[:-1]

    if show_sym:
        Uxy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Uyz = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        Uzx = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        t = np.zeros((Nb, Nb), dtype=complex)
        for R in np.unique(df[['i', 'j', 'k']], axis=0):
            df_R = df[(df['i'] == R[0]) & (df['j'] == R[1]) & (df['k'] == R[2])]
            for d in df_R.itertuples(index=False):
                t[d.p-1][d.q-1] += complex(d.t_real + d.t_imag * 1j)
        
        print('t =')
        for i in range(Nb):
            for j in range(Nb):
                print('%8.4f'%t[i][j].real, end='')
            print()
        print('=> Non-Hermitian' if np.count_nonzero(t - np.conjugate(t).T > 1e-6) else '=> Hermitian', '\n')

    if show_tR:
        df['t'] = np.sqrt(df['t_real']**2 + df['t_imag']**2)

        r1_norm = r_norm[1]
        t1_max = np.max(df[np.abs(df['norm']-r1_norm) < 1e-6]['t'])
        print('r1_norm =', r1_norm)
        print('t1_max =', t1_max, end='\n\n')

        print('%6s%16s%16s' % ('n', 'r_norm/r1_norm', 't_max/t1_max'))
        for i in range(1, len(r_norm)):
            t_max = np.max(df[np.abs(df['norm']-r_norm[i]) < 1e-6]['t'])
            print('%6d%16.6f%16.6f' % (i, r_norm[i]/r1_norm, t_max/t1_max))
        print()

    fn = 'lattice_n%d.txt' % n
    np.savetxt(fn, df[lat_cols], fmt=['%5d', '%5d', '%5d', '%5d', '%5d', '%12.6f', '%12.6f', '%12.6f'])
    print('File saved at %s\n' % fn)

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

def GenBand(path_lat, Nk, show_band=0):
    Nk, show_band = int(Nk), int(show_band)
    print('path_lat =', path_lat, '\nNk =', Nk, end='\n\n')

    df = pd.read_csv(path_lat, sep='\s+', names=lat_cols).astype(lat_dtypes)
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
    fn = '%s/band_%s_Nk%d.txt' % (band_path, re.sub('.txt', '', re.sub('lattice_', '', path_lat)), Nk)
    np.savetxt(fn, band)
    print('File saved at %s\n' % fn)
    if show_band: ShowBand(fn)

def ShowBand(path_band):
    path_band = path_band.split(':')
    print('path_band =', *path_band, sep='\n', end='\n\n')

    Nk = ReSubInt('Nk', path_band[0])
    k, k_label, k_point = GenK(Nk)
    for l, p in zip(k_label, k_point): print(l, p, end='\t')
    print(end='\n\n')

    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)
    color = plt.cm.tab10(range(10))
    ls = ['-', '--', '-.', ':'] * 2

    for i, p in enumerate(path_band):
        band, n = np.genfromtxt(p), ReSubInt('n', p)
        label = 'n=%d' % n 

        if i:
            label += ', $|E_{%d}-E_{%d}|_\mathrm{max}$=%.4f' % (n0, n, np.max(np.abs(band0 - band)))
            band_max, band_scale = np.max(band), np.max(band)-np.min(band)
            dband = band0_max - band_max; band += dband; #band *= band_scale/band0_scale
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

    fn = '%s/band_' % band_path + '_'.join([re.sub('_Nk.*', '', re.sub('%s/band_' % band_path, '', p)) for p in path_band]) + '_Nk%d.png' % Nk
    fig.savefig(fn)
    print('Figure saved at %s\n' % fn)
    plt.show()

if args.wann2lat: Wann2Lat(*args.wann2lat)
elif args.genband: GenBand(*args.genband)
elif args.showband: ShowBand(*args.showband)
else: parser.print_help()
