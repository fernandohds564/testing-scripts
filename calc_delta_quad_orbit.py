#!/usr/bin/env python-sirius

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.cm as cm

import pyaccel
from pymodels import si


np.random.seed(100022)
nrrand = 20

deltakl = 0.02

mod = si.create_accelerator()
mod.cavity_on = True
mod.radiation_on = True
famdata = si.get_family_data(mod)
twi, *_ = pyaccel.optics.calc_twiss(mod, indices='open')

print(twi.mux[-1]/2/np.pi, twi.muy[-1]/2/np.pi)

fam = 'QN'
bpm = np.array(famdata['BPM']['index']).flatten()
q1 = np.array(famdata[fam]['index']).flatten()[23]
is_skew = fam == 'QS'

KL = mod[q1].KL
nomx = -deltakl*np.sqrt(twi.betax[q1]*twi.betax) / 2 / np.sin(twi.mux[-1]/2)
nomx *= np.cos(np.abs(twi.mux-twi.mux[q1]) - twi.mux[-1]/2)
# nomx /= 1 + KL*twi.betax[q1]/np.tan(twi.mux[-1]/2)/2
nomx2 = np.dot(nomx[bpm], nomx[bpm])
nomy = deltakl*np.sqrt(twi.betay[q1]*twi.betay) / 2 / np.sin(twi.muy[-1]/2)
nomy *= np.cos(np.abs(twi.muy-twi.muy[q1]) - twi.muy[-1]/2)
# nomy /= 1 - KL*twi.betay[q1]/np.tan(twi.muy[-1]/2)/2
nomy2 = np.dot(nomy[bpm], nomy[bpm])
if is_skew:
    nomy = -nomy

qn = np.array(famdata['QN']['index']).flatten()
sn = np.array(famdata['SN']['index']).flatten()
idcs = [qn, sn]
errs = [7e-6, 10e-6]

x0s_neg, y0s_neg, x0s_ini, y0s_ini, x0s_pos, y0s_pos = [], [], [], [], [], []
dorbx, dorby, x0_calcd, y0_calcd = [], [], [], []
for i in range(nrrand):
    for idx, err in zip(idcs, errs):
        errx = 2*(np.random.rand(idx.size)-0.5)*err
        erry = 2*(np.random.rand(idx.size)-0.5)*err
        pyaccel.lattice.set_error_misalignment_x(mod, idx, errx)
        pyaccel.lattice.set_error_misalignment_y(mod, idx, erry)
    orb = pyaccel.tracking.find_orbit6(mod, indices='open')
    iniorbx = orb[0, :]
    iniorby = orb[2, :]
    x0 = (iniorbx[q1] + iniorbx[q1+1])/2
    y0 = (iniorby[q1] + iniorby[q1+1])/2
    x0s_ini.append(x0)
    y0s_ini.append(y0)
    if is_skew:
        mod[q1].KsL += deltakl/2
    else:
        mod[q1].KL += deltakl/2
    orb = pyaccel.tracking.find_orbit6(mod, indices='open')
    posorbx = orb[0, :]
    posorby = orb[2, :]
    x0 = (posorbx[q1] + posorbx[q1+1])/2
    y0 = (posorby[q1] + posorby[q1+1])/2
    x0s_pos.append(x0)
    y0s_pos.append(y0)
    if is_skew:
        mod[q1].KsL -= deltakl
    else:
        mod[q1].KL -= deltakl
    orb = pyaccel.tracking.find_orbit6(mod, indices='open')
    if is_skew:
        mod[q1].KsL += deltakl/2
    else:
        mod[q1].KL += deltakl/2
    negorbx = orb[0, :]
    negorby = orb[2, :]
    x0 = (negorbx[q1] + negorbx[q1+1])/2
    y0 = (negorby[q1] + negorby[q1+1])/2
    x0s_neg.append(x0)
    y0s_neg.append(y0)

    dorx = (posorbx-negorbx)
    dory = (posorby-negorby)
    # dorx = (posorbx-iniorbx)*2
    # dory = (posorby-iniorby)*2
    # dorx = (iniorbx-negorbx)*2
    # dory = (iniorby-negorby)*2
    dorbx.append(dorx)
    dorby.append(dory)

    x0_calcd.append(np.sum(nomx[bpm]*dorx[bpm])/nomx2)
    y0_calcd.append(np.sum(nomy[bpm]*dory[bpm])/nomy2)

x0s_ini = np.array(x0s_ini)
y0s_ini = np.array(y0s_ini)
x0s_neg = np.array(x0s_neg)
y0s_neg = np.array(y0s_neg)
x0s_pos = np.array(x0s_pos)
y0s_pos = np.array(y0s_pos)
dorbx = np.array(dorbx)
dorby = np.array(dorby)
x0_calcd = np.array(x0_calcd)
y0_calcd = np.array(y0_calcd)
if is_skew:
    x0_calcd, y0_calcd = y0_calcd, x0_calcd


# # Plot
f = plt.figure(figsize=(15, 10))
gs = mpl_gs.GridSpec(2, 3)
gs.update(
    left=0.10, right=0.97, top=0.97, bottom=0.05, hspace=0.4, wspace=0.25)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[0, 1])
ax4 = plt.subplot(gs[1, 1])
ax5 = plt.subplot(gs[0, 2])
ax6 = plt.subplot(gs[1, 2])
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax5.grid(True)
ax6.grid(True)

if is_skew:
    ls1 = ax1.plot(twi.spos, dorbx.T / y0s_ini[None, :])
    ls2 = ax2.plot(twi.spos, dorby.T / x0s_ini[None, :])
else:
    ls1 = ax1.plot(twi.spos, dorbx.T / x0s_ini[None, :])
    ls2 = ax2.plot(twi.spos, dorby.T / y0s_ini[None, :])
ax1.plot(twi.spos, nomx, label='nominal', color='k')
ax2.plot(twi.spos, nomy, label='nominal', color='k')

ax3.plot(np.std(dorbx[:, bpm]*1e6, axis=1))
ax4.plot(np.std(dorby[:, bpm]*1e6, axis=1))

colors = cm.brg(np.linspace(0, 1, x0s_ini.size))
axx = ax6 if is_skew else ax5
axy = ax5 if is_skew else ax6
for i in range(x0s_ini.size):
    ls1[i].set_color(colors[i])
    ls2[i].set_color(colors[i])
    axx.plot(i, x0s_ini[i]*1e6, 'o', color=colors[i])
    axy.plot(i, y0s_ini[i]*1e6, 'o', color=colors[i])
    axx.plot(i, x0s_pos[i]*1e6, '+', color=colors[i])
    axy.plot(i, y0s_pos[i]*1e6, '+', color=colors[i])
    axx.plot(i, x0s_neg[i]*1e6, 'x', color=colors[i])
    axy.plot(i, y0s_neg[i]*1e6, 'x', color=colors[i])


axx.plot(x0_calcd*1e6, color='orange')
axy.plot(y0_calcd*1e6, color='orange')
plt.show()
