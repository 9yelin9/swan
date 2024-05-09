from . import env

import os
import re
import ctypes
import cycler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Swann:
    def __init__(self):
        np.set_printoptions(precision=6, suppress=True)
        plt.rcParams.update({'font.size': 28})
        self.Dim = 3

    def ReadWin(self):
        self.Nb, self.Nc, self.Ni = 0, 0, 0
        self.k_path, self.pos, self.basis, self.A = [], [], [], []

        with open('%s/wannier90.win' % self.path_data, 'r') as f:
            read_line = 0
            for line in f:
                if re.search('end kpoint_path', line): break
                elif read_line: self.k_path.append(line.split())
                elif re.search('begin kpoint_path', line): read_line = 1

            read_line = 0
            for line in f:
                if re.search('end projections', line): break
                elif read_line:
                    self.pos.append(re.sub('f=|:.+', '', line).split(','))
                    self.basis.append(re.sub('.+:|\n', '', line))
                elif re.search('begin projections', line): read_line = 1
            self.Nb, self.Nc = len(self.basis), len(set(self.basis)); self.Ni = self.Nb // self.Nc
            self.pos = np.round(np.array(self.pos, dtype='d'), decimals=6)[::self.Nc]

            read_line = 0
            for line in f:
                if re.search('end unit_cell_cart', line): break
                elif read_line: self.A.append(line.split())
                elif re.search('begin unit_cell_cart', line): read_line = 1
            self.A = np.array(self.A, dtype='d'); self.A /= self.A[self.A > 0][0]
            self.pos = np.array([np.sum([p[i]*self.A[i] for i in range(self.Dim)], axis=0) for p in self.pos])

        print('Nb =', self.Nb, 'Nc =', self.Nc, 'Ni =', self.Ni)
        print('path =', *self.k_path, sep='\n')
        print('pos =', *self.pos, sep='\n')
        print('A =', *self.A, sep='\n', end='\n\n')

    def GetDirSave(self, path_in, path_out):
        self.path_data = path_in.split('swann')[0]
        dn = '%s/%s' % (self.path_data, path_out)
        os.makedirs(dn, exist_ok=True)
        return dn

    def GetLat(self, path_lat):
        df = pd.read_csv(path_lat, sep='\s+', names=env.lat_col).astype(dict(list(env.lat_dtype.items())))

        r_list = np.array([np.sum([d[i]*self.A[i] for i in range(self.Dim)], axis=0) + self.pos[(d.q-1)//self.Nc] - self.pos[(d.p-1)//self.Nc] for d in df.itertuples(index=False)])
        norm_list = np.round(np.linalg.norm(r_list, axis=1), decimals=6)
        df['r1'], df['r2'], df['r3'], df['norm'] = *r_list.T, norm_list; norm_list = np.unique(norm_list)
        df.sort_values(by=['norm', 'i', 'j', 'k', 'p', 'q'], inplace=True)

        return df, r_list, norm_list

    def GetT(self, df):
        t_list = []
        for norm, df_norm in df.groupby('norm'):
            for r, df_norm_r in df_norm.groupby(['r1', 'r2', 'r3']):
                t = np.zeros((self.Nc, self.Nc), dtype=complex)
                for d in df_norm_r.itertuples(index=False):
                    t[(d.p-1)%self.Nc][(d.q-1)%self.Nc] += complex(d.t_real + d.t_imag * 1j)
                t_list.append([norm, r, t])

        return t_list

    def GetK(Nk):
        path_label, path_point = [], []
        for p in k_path:
            path_label.append([p[0],       p[self.Dim+1]])
            path_point.append([p[1:self.Dim+1], p[self.Dim+2:]])
        path_label, path_point = np.array(path_label), np.array(path_point, dtype='d')

        k_label = np.append(path_label[:, 0], path_label[-1, 1])

        dist = []
        for pi, pf in path_point:
            ki, kf = np.zeros(self.Dim), np.zeros(self.Dim)
            for i in range(self.Dim):
                ki = np.add(ki, pi[i]*self.A[i])
                kf = np.add(kf, pf[i]*self.A[i])
            dist.append(np.linalg.norm(ki - kf))
        dist = [int((Nk-1)*d / sum(dist)) for d in dist]
        dist[-1] += (Nk-1) - sum(dist)
        k_point = [0] + list(np.cumsum(dist))

        k = []
        for (pi, pf), d in zip((2*np.pi)*path_point, dist):
            Nd = d
            if d == dist[-1]: Nd = d + 1
            for di in range(Nd): k.append([pi[i] + (pf[i] - pi[i]) * di/d for i in range(self.Dim)])

        return k, k_label, k_point

    def GenLat(self, path_data, n, show_t=0, show_tR=0):
        dn = self.GetDirSave(path_data, env.path_lat); self.ReadWin()
        df, r_list, norm_list = self.GetLat('%s/lattice_n0.txt' % dn)

        if n:
            norm_list = np.unique(norm_list)[:n+2] #r_list = r_list[norm_list < norm_list[n+1]]; r_list_uq = np.unique(r_list, axis=0)
            df = df[df['norm'] < norm_list[n+1]]
            norm_list = norm_list[:-1]

        if show_t:
            t_list = self.GetT(df)
            for norm, r, t in t_list:
                print(norm, r, '\n', t.real, end='\n\n')

        if show_tR:
            df['t'] = np.sqrt(df['t_real']**2 + df['t_imag']**2)

            t1_max = np.max(df[np.abs(df['norm']-norm_list[1]) < 1e-6]['t'])
            print('r1_norm =', norm_list[1])
            print('t1_max =', t1_max, end='\n\n')

            print('%6s%16s%16s' % ('n', 'r_norm/r1_norm', 't_max/t1_max'))
            for i in range(1, len(norm_list)):
                t_max = np.max(df[np.abs(df['norm']-norm_list[i]) < 1e-6]['t'])
                print('%6d%16.6f%16.6f' % (i, norm_list[i]/norm_list[1], t_max/t1_max))
            print()

        fn = '%s/lattice_n%d.txt' % (dn, n)
        np.savetxt(fn, df[env.lat_col], fmt=env.lat_format)
        print('File saved at %s\n' % fn)

    def GenBand(self, path_lat, Nk, show_band=0):
        dn = self.GetDirSave(path_lat, env.path_band); self.ReadWin()
        Nk, show_band = int(Nk), int(show_band)
        print('path_lat =', path_lat, '\nNk =', Nk, end='\n\n')

        df, _, _ = self.GetLat(path_lat) 
        k, k_label, k_point = self.GetK(Nk)
        if self.Nb != df['p'].max(): print('Wrong Nb =', df['p'].max(), end='\n\n'); sys.exit(1)

        site_c = np.ctypeslib.as_ctypes(np.ravel(df[['i', 'j', 'k']].astype(dtype='i')))
        obt_c  = np.ctypeslib.as_ctypes(np.ravel(df[['p', 'q']].astype(dtype='i')))
        t_c    = np.ctypeslib.as_ctypes(np.ravel(df[['t_real', 't_imag']]))
        k_c    = np.ctypeslib.as_ctypes(np.ravel(k))
        band_c = np.ctypeslib.as_ctypes(np.ravel(np.zeros((Nk, self.Nb))))

        swann_c = ctypes.cdll.LoadLibrary('%s/libswann.so' % env.path_swann)
        swann_c.GenBand(env.num_thread, self.Dim, self.Nb, Nk, len(df), k_c, site_c, obt_c, t_c, band_c)
        band = np.reshape(np.ctypeslib.as_array(band_c), (Nk, self.Nb))

        fn = '%s/%s' % (dn, re.sub('lattice_', 'band_', re.sub('[.]txt', '_Nk%d.txt' % Nk, path_lat)))
        np.savetxt(fn, band)
        print('File saved at %s\n' % fn)
        if show_band: ShowBand(fn)

    def ShowT(self, path_lat):
        self.GetDirSave(path_lat, env.path_lat); self.ReadWin()
        t_list = self.GetT(self.GetLat(path_lat)[0])

        not_hermitian = []
        for i, (norm, r, t) in enumerate(t_list):
            if np.count_nonzero(np.abs(t - np.conjugate(t).T) > 1e-6): not_hermitian.append(i)
            print(norm, r, '\n', t, end='\n\n')

        print('%d of Non-hermitian t:' % len(not_hermitian))
        for t_idx in not_hermitian:
            print(norm, r, '\n', t, end='\n\n')
        
    def ShowBand(self, path_band, fit_point=0):
        path_band = path_band.split(':'); dn = self.GetDirSave(path_band[0], env.path_fig)
        print('path_band =', *path_band, sep='\n', end='\n\n')

        Nk = ReSubInt('Nk', path_band[0])
        k, k_label, k_point = self.GetK(Nk)
        fit_point, fit_label = int(fit_point), ''
        for l, p in zip(k_label, k_point):
            if fit_point <= p: fit_label = l
            print(l, p, end='\t')
        print(end='\n\n')

        fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)
        color = plt.cm.tab10(range(10))
        #ls = ['-', '--', '-.', ':'] * 2

        for i, p in enumerate(path_band):
            band, n = np.genfromtxt(p), ReSubInt('n', p)
            label = 'n=%d' % n 

            if fit_point:
                point = band[fit_point]
                band_scale = np.max(point)-np.min(point)
                if i: band *= band0_scale/band_scale; band += band0_max - np.max(point); label += ' (fitted at %s)' % fit_label
                else: band0_scale, band0_max = band_scale, np.max(point)

            ax.plot(range(Nk), band[:, 0], color=color[i], label=label)
            for j in range(1, band.shape[1]): ax.plot(range(Nk), band[:, j], color=color[i])
        ax.grid(True, axis='x')
        ax.legend(fontsize='xx-small')
        ax.set_ylim([np.min(band)-0.5, np.max(band)+0.5])
        ax.set_xticks(k_point, labels=k_label)
        ax.set_ylabel(r'$E$')

        fn = '%s/band_' % (dn + '_'.join([re.sub('%s/band_' % env.path_save, '', re.sub('_Nk.*', '', p)) for p in path_band]) + '_Nk%d.png' % Nk)
        fig.savefig(fn)
        print('Figure saved at %s\n' % fn)
        plt.show()
