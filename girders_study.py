#!/usr/bin/env python-sirius

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.cm as cm

import pyaccel
from pymodels import si
from pymodels.middlelayer.devices import SOFB
from siriuspy.clientconfigdb import ConfigDBClient
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat


def get_correctors_strength():
    clt_gbl = ConfigDBClient(config_type='global_config')
    conf = clt_gbl.get_config_value('energy2p963gev_delta_quads_bo_config_A')
    conf = {n: v for n, v, d in conf['pvs']}

    vals = []
    for name in sofb.data.CH_NAMES:
        vals.append(conf[name + ':Current-SP'])
    return np.array(vals) * 39.0969 *1e-6


def correct_orbit(mod, irm, bpm, ch, cv, rf):
    respmcalc = OrbRespmat(model, 'SI', dim='6d')
    respmcalc.bpms = bpm
    for i in range(5):
        respmcalc.model = mod
        respmat = respmcalc.get_respm()
        u, s, vh = np.linalg.svd(respmat, full_matrices=False)
        invs = 1/s
        invs[-20:] = 0
        irespmat = vh.T @ np.diag(invs) @ u.T

        orb = pyaccel.tracking.find_orbit6(mod, indices=bpm)
        orbx, orby = orb[0], orb[2]
        print(i, np.std(orbx), np.mean(orbx))
        corrs = -irespmat @ np.hstack([orbx, orby])
        chkick = pyaccel.lattice.get_attribute(mod, 'hkick_polynom', ch)
        cvkick = pyaccel.lattice.get_attribute(mod, 'vkick_polynom', cv)
        freq = pyaccel.lattice.get_attribute(mod, 'frequency', rf)
        chkick += corrs[:120]
        cvkick += corrs[120:280]
        freq += corrs[280]
        pyaccel.lattice.set_attribute(mod, 'hkick_polynom', ch, chkick)
        pyaccel.lattice.set_attribute(mod, 'vkick_polynom', cv, cvkick)
        pyaccel.lattice.set_attribute(mod, 'frequency', rf, freq)
    print('')


def add_girder_errorx(mod, girds, girtyp, err):
    for gird, idx in girds.items():
        if not gird.endswith(girtyp):
            continue
        pyaccel.lattice.set_error_misalignment_x(mod, idx, err)


sofb = SOFB('SI')
model = si.create_accelerator()
model.cavity_on = True
# model.radiation_on = True
respmcalc = OrbRespmat(model, 'SI', dim='6d')
bpms = respmcalc.bpms
chs = respmcalc.ch
cvs = respmcalc.cv
rf = pyaccel.lattice.find_indices(model, 'pass_method', 'cavity_pass')
freq0 = pyaccel.lattice.get_attribute(model, 'frequency', rf)

respmat = respmcalc.get_respm()
u, s, vh = np.linalg.svd(respmat, full_matrices=False)
irespmat = vh.T @ np.diag(1/s) @ u.T

girders = respmcalc.fam_data['girder']
girs = dict()
for i, sub in enumerate(girders['subsection']):
    name = sub + girders['instance'][i]
    girs[name] = girders['index'][i]
girders = girs

gird_tp = sorted(set([g[2:] for g in girders.keys()]))
# gird_tp = ['C12', 'M2', 'C21', 'C41']

convgird_tp = {
    'C11': 'B11', 'C12': 'C1', 'C21': 'B21', 'C22': 'C2', 'C31': 'C3',
    'C32': 'B22', 'C41': 'C4', 'C42': 'B12', 'M1': 'M1', 'M2': 'M2',
    'BC': 'BC', 'B1B2': 'B1B2', 'B1C': 'B1C', 'B2C': 'B2C'}

results = dict()
errx = 40e-6
mat = []
for gtp in gird_tp:
    res = dict()

    add_girder_errorx(model, girders, gtp, errx)
    correct_orbit(model, irespmat, bpms, chs, cvs, rf)
    res['ch'] = np.array(
        pyaccel.lattice.get_attribute(model, 'hkick_polynom', chs))
    res['cv'] = np.array(
        pyaccel.lattice.get_attribute(model, 'vkick_polynom', cvs))
    results[gtp] = res
    mat.append(res['ch'])

    add_girder_errorx(model, girders, gtp, 0)
    pyaccel.lattice.set_attribute(model, 'hkick_polynom', chs, 0)
    pyaccel.lattice.set_attribute(model, 'vkick_polynom', cvs, 0)
    pyaccel.lattice.set_attribute(model, 'frequency', rf, freq0)

mat = np.array(mat).T / errx
u, s, vh = np.linalg.svd(mat, full_matrices=False)
invs = 1/s
invs[7:] = 0
imat = vh.T @ np.diag(invs) @ u.T

confv = get_correctors_strength()
confv = sofb.kickch * 1e-6

errors = imat @ confv

corrs = []
relation = []
c2 = np.dot(confv, confv)
for i in range(mat.shape[1]):
    m = mat[:, i]
    corrs.append(np.dot(confv, m) / np.dot(m, m))
    relation.append(np.dot(confv, m)**2 / np.dot(m, m) / c2)
for gtp, err, corr, rel in zip(gird_tp, errors, corrs, relation):
    print('{:5s}: {:-6.1f} {:6.1f} {:4.1f}'.format(
        convgird_tp[gtp], err*1e6, corr*1e6, rel*100))

fit = mat @ errors

f = plt.figure(figsize=(15, 10))
gs = mpl_gs.GridSpec(1, 1)
gs.update(
    left=0.10, right=0.97, top=0.97, bottom=0.05, hspace=0.4, wspace=0.25)
ax = plt.subplot(gs[0, 0])
ax.grid(True)

# for gtp, res in results.items():
#     ax.plot(-2*res['ch']*1e6, label=convgird_tp[gtp])

print('std real', np.std(confv)*1e6)
print('std diff', np.std(confv - fit)*1e6)

ax.plot(confv*1e6, '-k', label='Before Correction')
ax.plot(fit*1e6, 'o--', label='Correction')
ax.plot((confv - fit)*1e6, label='Predicted After Correction')
ax.legend()

# ##### FFT #####
# f = plt.figure(figsize=(15, 10))
# gs = mpl_gs.GridSpec(1, 1)
# gs.update(
#     left=0.10, right=0.97, top=0.97, bottom=0.05, hspace=0.4, wspace=0.25)
# ax = plt.subplot(gs[0, 0])
# ax.grid(True)

# fft = np.fft.rfft(confv)
# freq = np.fft.rfftfreq(confv.size, d=1/120)
# ax.plot(freq, np.abs(fft), '-k', label='CH Strength')
# ax.plot(freq, np.abs(np.fft.rfft(confv - fit)), label='diff')
# ax.legend()

corrmat = mat.T @ mat
diag = np.diag(corrmat)
corrmat = corrmat*corrmat / (diag[:, None] @ diag[None, :])

f = plt.figure(figsize=(15, 10))
gs = mpl_gs.GridSpec(1, 1)
gs.update(
    left=0.10, right=0.97, top=0.97, bottom=0.05, hspace=0.4, wspace=0.25)
ax = plt.subplot(gs[0, 0])
ax.grid(True)
tri = np.tri(corrmat.shape[0], k=0)
ax.plot(tri*corrmat)
ax.legend(gird_tp)

plt.show()
