#!/usr/gin/env python-sirius

import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs
from matplotlib import rcParams

from pymodels import si
import pyaccel

rcParams.update({'lines.linewidth': 2, 'axes.grid': True, 'font.size': 14})


def fit_curve(pos, data, nharms=5, lamb0=518.396):
    mat = [np.ones(data.size), ]
    for i in range(1, nharms+1):
        mat.append(np.sin(2*np.pi*pos/lamb0*i))
        mat.append(np.cos(2*np.pi*pos/lamb0*i))

    mat = np.array(mat).T
    coeff, *_ = np.linalg.lstsq(mat, data, rcond=None)
    # u, s, vt = np.linalg.svd(mat, full_matrices=False)
    # coeff = (vt.T/s @ u.T) @ data
    curve = mat@coeff
    return curve


pos, hor, prec_h, ver, prec_v, lon, prec_l, roll, prec_r = np.loadtxt(
    'alignment_data.txt', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    unpack=True)

model = si.create_accelerator()
famdata = si.get_family_data(model)
idx = famdata['girder']['subsection'].index('17C1')
idx = famdata['girder']['index'][idx][0]
model = pyaccel.lattice.shift(model, idx)
model.cavity_on = True
twiss, *_ = pyaccel.optics.calc_twiss(model)

mux = np.interp(pos, twiss.spos, twiss.mux)
muy = np.interp(pos, twiss.spos, twiss.muy)

fig = mplt.figure(figsize=(8, 5))
gs = mgs.GridSpec(2, 1)
gs.update(left=0.12, right=0.8, top=0.95, bottom=0.1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

for nharms in range(5, 10):
    fit_h = fit_curve(mux, hor, nharms=nharms, lamb0=twiss.mux[-1])
    fit_v = fit_curve(muy, ver, nharms=nharms, lamb0=twiss.muy[-1])
    ax1.plot(pos, fit_h, label=f'{nharms}')
    ax2.plot(pos, fit_v, label=f'{nharms}')

ax1.plot(pos, hor, label='Data')
ax2.plot(pos, ver, label='Data')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0), title='# Harmonics')

ax1.set_ylabel('Horizontal [mm]')
ax2.set_ylabel('Vertical [mm]')
ax2.set_xlabel('Position [m]')
mplt.show()
