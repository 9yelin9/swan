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

        self.num_thread = 1
        self.Dim = 3

        self.path_swann = '/home/9yelin9/R2Ir2O7/hf3/swann'
        self.lat_col = ['i', 'j', 'k', 'p', 'q', 't_real', 't_imag']
        self.lat_dtype = {'i':'i', 'j':'i', 'k':'i', 'p':'i', 'q':'i', 't_real':'d', 't_imag':'d'}
        self.lat_format = ['%5d', '%5d', '%5d', '%5d', '%5d', '%12.6f', '%12.6f']

    def ReSubFloat(self, pattern, string):
        return float(re.sub(pattern, '', re.search('%s[-]?\d+[.]\d+' % pattern, string).group()))

    def ReSubInt(self, pattern, string):
        return int(re.sub(pattern, '', re.search('%s[-]?\d+' % pattern, string).group()))

    def ReadWin(self, path, dir_save):
        self.path_data = path.split('swann')[0]
        self.path_save = '%s/swann/%s' % (self.path_data, dir_save)
        os.makedirs(self.path_save, exist_ok=True)

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

        """
        print('Nb =', self.Nb, 'Nc =', self.Nc, 'Ni =', self.Ni)
        print('path =', *self.k_path, sep='\n')
        print('pos =', *self.pos, sep='\n')
        print('A =', *self.A, sep='\n', end='\n\n')
        """

    def GetK(self, Nk):
        path_label, path_point = [], []
        for p in self.k_path:
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

    def AddNorm(self, df):
        r_list = np.array([np.sum([d[i]*self.A[i] for i in range(self.Dim)], axis=0) + self.pos[(d.q-1)//self.Nc] - self.pos[(d.p-1)//self.Nc] for d in df.itertuples(index=False)])
        norm_list = np.round(np.linalg.norm(r_list, axis=1), decimals=6)
        df['r1'], df['r2'], df['r3'], df['norm'] = *r_list.T, norm_list;
        df.sort_values(by=['norm', 'i', 'j', 'k', 'p', 'q'], inplace=True); df.reset_index(drop=True, inplace=True)
        return df

    def GenLat(self, path_data, n, show_tR=0, show_df=0):
        n, show_tR = int(n), int(show_tR)
        self.ReadWin(path_data, 'lat')

        pat_site = '[-]?\d+\s+'
        pat_obt  = '[-]?\d+\s+'
        pat_t    = '[-]?\d+[.]\d+\s+'
        pat = self.Dim * pat_site + 2 * pat_obt + 2 * pat_t

        with open('%s/wannier90_hr.dat' % path_data, 'r') as f:
            fp, line = 0, f.readline()
            while line:
                if re.search(pat, line): break
                else: fp, line = f.tell(), f.readline()
            f.seek(fp)
            df = pd.read_csv(f, sep='\s+', names=self.lat_col).astype(dict(list(self.lat_dtype.items()))); df = self.AddNorm(df)

        if n:
            norm_list = np.unique(df['norm'])[:n+2]
            df = df[df['norm'] < norm_list[n+1]]; df.reset_index(drop=True, inplace=True)
            norm_list = norm_list[:-1]

        if show_tR:
            df['t'] = np.sqrt(df['t_real']**2 + df['t_imag']**2)
            t1_max = np.max(df[np.abs(df['norm']-norm_list[1]) < 1e-6]['t'])
            print('r1_norm =', norm_list[1])
            print('t1_max =', t1_max, end='\n\n')

            print('%6s%16s%16s' % ('n', 'r_norm/r1_norm', 't_max/t1_max'))
            for i in range(1, len(norm_list)):
                t_max = np.max(df[np.abs(df['norm']-norm_list[i]) < 1e-6]['t'])
                print('%6d%16.6f%16.6f' % (i, norm_list[i]/norm_list[1], t_max/t1_max))
            print(); df.drop(columns=['t'], inplace=True)

        if show_df: print(df)

        fn = '%s/lat_n%d.h5' % (self.path_save, n)
        df.to_hdf(fn, key='lat', mode='w')
        print('File saved at %s\n' % fn)

    def GenBand(self, path_lat, Nk, show_band=0):
        Nk, show_band = int(Nk), int(show_band)
        self.ReadWin(path_lat, 'band')
        print('path_lat =', path_lat, '\nNk =', Nk, end='\n\n')

        df = pd.read_hdf(path_lat, key='lat') 
        k, k_label, k_point = self.GetK(Nk)
        if self.Nb != df['p'].max(): print('Wrong Nb =', df['p'].max(), end='\n\n'); sys.exit(1)

        site_c = np.ctypeslib.as_ctypes(np.ravel(df[['i', 'j', 'k']].astype(dtype='i')))
        obt_c  = np.ctypeslib.as_ctypes(np.ravel(df[['p', 'q']].astype(dtype='i')))
        t_c    = np.ctypeslib.as_ctypes(np.ravel(df[['t_real', 't_imag']]))
        k_c    = np.ctypeslib.as_ctypes(np.ravel(k))
        band_c = np.ctypeslib.as_ctypes(np.ravel(np.zeros((Nk, self.Nb))))

        swann_c = ctypes.cdll.LoadLibrary('%s/libswann.so' % self.path_swann)
        swann_c.GenBand(self.num_thread, self.Dim, self.Nb, Nk, len(df), k_c, site_c, obt_c, t_c, band_c)
        band = np.reshape(np.ctypeslib.as_array(band_c), (Nk, self.Nb))

        fn = '%s' % re.sub('lat', 'band', re.sub('[.]h5', '_Nk%d.txt' % Nk, path_lat))
        np.savetxt(fn, band)
        print('File saved at %s\n' % fn)
        if show_band: ShowBand(fn)

        return fn

    def ShowBand(self, path_band, fit_point=0):
        path_band = path_band.split(':'); self.ReadWin(path_band[0], 'fig')
        print('path_band =', *path_band, sep='\n', end='\n\n')

        Nk = self.ReSubInt('Nk', path_band[0])
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
            band, n = np.genfromtxt(p), self.ReSubInt('n', p)
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

        fn = '%s/band_%s' % (self.path_save, '_'.join([re.sub('.*band_', '', re.sub('_Nk.*', '', p)) for p in path_band]) + '_Nk%d.png' % Nk)
        fig.savefig(fn)
        print('Figure saved at %s\n' % fn)
        plt.show()
