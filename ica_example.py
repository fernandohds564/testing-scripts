#!/usr/bin/env python-sirius

import numpy as np
# import scipy.optimize as scyopt

from sklearn.decomposition import FastICA
# from sklearn.model_selection import GridSearchCV
# from lightning.classification import FistaClassifier

import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs
from matplotlib import rcParams
import seaborn as sns

rcParams.update({'lines.linewidth': 2, 'axes.grid': True, 'font.size': 14})


# f = 10
# t = np.linspace(0, 1, 1000) * 2*np.pi
# dt = t[1] - t[0]

# y = np.sin(f*t)
# yl = f*np.cos(f*t)

# yn = y + np.random.randn(t.size)*0.01

# ynl0 = np.diff(yn)/np.diff(t)
# ynl1 = np.gradient(yn, dt, edge_order=2)

# freqs = np.fft.rfftfreq(t.size, d=dt)
# ynl2 = np.fft.irfft(1j*2*np.pi*freqs*np.fft.rfft(yn))

# fista = FistaClassifier(penalty='tv1d')
# gs = GridSearchCV(fista, {'alpha': np.logspace(-3, 3, 10)})
# gs.fit(np.eye(yn.size), yn)
# yr = gs.best_estimator_.coef_

# f = mplt.figure(figsize=(7, 7))
# gs = mgs.GridSpec(2, 1)

# ax1 = f.add_subplot(gs[0, 0])
# ax2 = f.add_subplot(gs[1, 0])

# ax1.plot(t, y, label='Real')
# ax1.plot(t, yn, label='Noised')
# ax1.plot(t, yr, label='TV')
# ax1.legend(loc='best')

# ax2.plot(t[:-1], ynl0, label='Diff')
# ax2.plot(t, ynl1, label='Grad')
# ax2.plot(t, ynl2, label='FFT')
# ax2.plot(t, yl, label='Real')
# ax2.legend(loc='best')

# mplt.show()

# #### Create Original signals
t = np.linspace(0, 10, 1000)
sig = []
sig.append(np.sin(4*t))
# sig.append(2*np.sin(8*t))
# sig.append(3*np.sin(6*t))
sig.append(1.0 * ((t > 3) & (t < 7)))
# sig.append(2.0 * ((t > 5) & (t < 9)))
sig.append(np.arcsin(np.sin(3*t)))
sig = np.array(sig)

# #### Mix data
np.random.seed(1029084)
A = np.random.rand(sig.shape[0] + 1, sig.shape[0])
x = A @ sig
# x += 0.1*np.random.randn(*x.shape)

xm = x-x.mean(axis=1)[:, None]

# #### Perform PCA
u, s, v = np.linalg.svd(xm, full_matrices=False)
nm = np.sum(s > 1e-6)
s_pca = v[:nm, :].T*s[:nm]

# #### Apply ICA
xw = (u/s).T[:sig.shape[0]] @ xm
xw *= np.sqrt(xm.shape[1])
ica = FastICA(algorithm='deflation', tol=1e-12)
ica.whiten = True
ica.n_components = sig.shape[0]
s_ica = ica.fit_transform(xm.T)
# ica.whiten = False
# s_ica = ica.fit_transform(xw.T)
s_ica *= s[:nm].mean()

# #### Plot data
f = mplt.figure(figsize=(10, 10))
gs = mgs.GridSpec(4, 3)
gs.update(
    top=0.96, bottom=0.05, left=0.08, right=0.98, hspace=0.3, wspace=0.35)

ax1a = f.add_subplot(gs[0, :2])
ax1b = f.add_subplot(gs[0, 2])
ax2a = f.add_subplot(gs[1, :2])
ax2b = f.add_subplot(gs[1, 2])
ax3a = f.add_subplot(gs[2, :2])
ax3b = f.add_subplot(gs[2, 2])
ax4a = f.add_subplot(gs[3, :2])
ax4b = f.add_subplot(gs[3, 2])

ax1a.set_title('Hidden Signal')
ax1a.plot(t, sig.T)
for xi in sig:
    sns.kdeplot(xi, ax=ax1b)

ax2a.set_title('Measured Data')
ax2a.plot(t, x.T)
for xi in x:
    sns.kdeplot(xi, ax=ax2b)

ax3a.set_title('PCA Recovered Signal')
ax3a.plot(t, s_pca)
for xi in s_pca.T:
    sns.kdeplot(xi, ax=ax3b)

ax4a.set_title('ICA Recovered Signal')
ax4a.plot(t, s_ica)
for xi in s_ica.T:
    sns.kdeplot(xi, ax=ax4b)

mplt.show()
