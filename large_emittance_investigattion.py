#!/usr/gin/env python-sirius

import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs
from matplotlib import rcParams

from pymodels import si
import pyaccel

rcParams.update({'lines.linewidth': 2, 'axes.grid': True, 'font.size': 14})


def find_drift(model, idx, forward=True):
    incr = 1 if forward else -1
    while idx >= 0 and model[idx].pass_method != 'drift_pass':
        idx += incr
    return idx


def shift_section(model, idx1, idx2, delta_s):
    idx1_n = find_drift(model, idx1, forward=False)
    idx2_n = find_drift(model, idx2, forward=True)
    model[idx1_n].length += delta_s
    model[idx2_n].length -= delta_s


def increase_distance(model, idx1, idx2, delta_s):
    shift_section(model, idx1, idx1, -delta_s/2)
    shift_section(model, idx2, idx2, delta_s/2)


def quadrupoles_distance_hypotesis(model, delta_s):
    # quadrupoles hypotesis
    q1i = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'Q1'))
    q2i = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'Q2'))
    q3i = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'Q3'))
    q4i = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'Q4'))
    # In sections C3 and C4 Q3 comes before Q4 and Q2 comes before Q1:
    bckup = q1i.copy()
    q1i[1::2] = q2i[1::2]
    q2i[1::2] = bckup[1::2]
    bckup = q3i.copy()
    q3i[1::2] = q4i[1::2]
    q4i[1::2] = bckup[1::2]

    # dipoles hypotesis
    # idcs = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'B1'))
    # idcs = idcs.reshape(40, -1)
    # q1i = idcs[:, 0]
    # q2i = idcs[:, -1]
    # idcs = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'B2'))
    # idcs = idcs.reshape(40, -1)
    # q3i = idcs[:, 0]
    # q4i = idcs[:, -1]

    drifts = np.ones((len(q1i), 2)) * delta_s
    # drifts = (np.random.rand(len(q1i), 2) - 0.5) * delta_s
    # drifts[:, 1] = 0

    for i, (q1, q2, q3, q4, drt) in enumerate(zip(q1i, q2i, q3i, q4i, drifts)):
        increase_distance(model, q1, q2, drt[0])
        increase_distance(model, q3, q4, drt[1])
        # shift_section(model, q1, q2, drt[0])
        # shift_section(model, q3, q4, drt[1])

    return pyaccel.optics.EquilibriumParameters(model)


mod = si.create_accelerator()
mod = pyaccel.lattice.refine_lattice(
    mod, max_length=0.02, pass_methods=['bnd_mpole_symplectic4_pass', ])

drift = 10e-3

eqpar1 = pyaccel.optics.EquilibriumParameters(mod)
print(eqpar1)
eqpar2 = quadrupoles_distance_hypotesis(mod, drift)
print(eqpar2)

f = mplt.figure(figsize=(8, 8))
gs = mgs.GridSpec(3, 1)
gs.update(left=0.12, right=0.98, top=0.95, bottom=0.12, hspace=0.02)

ax1 = f.add_subplot(gs[0, 0])
ax2 = f.add_subplot(gs[1, 0], sharex=ax1)
ax3 = f.add_subplot(gs[2, 0], sharex=ax1)

ax1.plot(eqpar1.twiss.spos, eqpar1.twiss.betax)
ax1.plot(eqpar2.twiss.spos, eqpar2.twiss.betax)
ax2.plot(eqpar1.twiss.spos, eqpar1.twiss.betay)
ax2.plot(eqpar2.twiss.spos, eqpar2.twiss.betay)
ax3.plot(eqpar1.twiss.spos, eqpar1.twiss.etax)
ax3.plot(eqpar2.twiss.spos, eqpar2.twiss.etax)
ax1.set_xlim([0, mod.length/5])

ax1.set_title(f'Displacement of {drift*1e3:0.1f} mm')
ax1.set_ylabel(r'$\beta_x$ [m]')
ax2.set_ylabel(r'$\beta_y$ [m]')
ax3.set_ylabel(r'$\eta_x$ [m]')
ax3.set_xlabel('Position [m]')

mplt.show()
