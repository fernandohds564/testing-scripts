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

# Remove C2
# c2 = bpms[3::8]
# bpms = sorted(set(bpms) - set(c2))
# respmcalc.bpms = bpms

respmat = respmcalc.get_respm()
u, s, vh = np.linalg.svd(respmat, full_matrices=False)
irespmat = vh.T @ np.diag(1/s) @ u.T

results = dict()
err = 10e-6
mat = []
fams = ['B1', 'B2', 'BC']
fams = ['B1', 'B2']
# fams = ['B1B2', ]
# fams = ['BC', ]
angs = []
for fam in fams:
    res = dict()
    if fam == 'B1B2':
        b1 = np.array(respmcalc.fam_data['B1']['index']).flatten()
        b2 = np.array(respmcalc.fam_data['B2']['index']).flatten()
        b1 = np.hstack([b1, b2]).flatten()
    else:
        b1 = np.array(respmcalc.fam_data[fam]['index']).flatten()
    ang1 = np.array(pyaccel.lattice.get_attribute(model, 'angle', b1))
    kl1 = np.array(pyaccel.lattice.get_attribute(model, 'KL', b1))
    ndip = 20 if fam == 'BC' else 40
    kick = ang1/np.sum(ang1)*ndip * err
    angs.append(np.sum(ang1)/ndip)

    pyaccel.lattice.set_attribute(model, 'hkick_polynom', b1, kick)
    # pyaccel.lattice.set_attribute(model, 'KL', b1, kl1*(1+err))

    correct_orbit(model, irespmat, bpms, chs, cvs, rf)
    res['ch'] = np.array(
        pyaccel.lattice.get_attribute(model, 'hkick_polynom', chs))
    res['cv'] = np.array(
        pyaccel.lattice.get_attribute(model, 'vkick_polynom', cvs))
    results[fam] = res
    mat.append(res['ch'])

    pyaccel.lattice.set_attribute(model, 'hkick_polynom', b1, 0)
    pyaccel.lattice.set_attribute(model, 'KL', b1, kl1)
    pyaccel.lattice.set_attribute(model, 'hkick_polynom', chs, 0)
    pyaccel.lattice.set_attribute(model, 'vkick_polynom', cvs, 0)
    pyaccel.lattice.set_attribute(model, 'frequency', rf, freq0)

mat = np.array(mat).T / err
u, s, vh = np.linalg.svd(mat, full_matrices=False)
invs = 1/s
imat = vh.T @ np.diag(invs) @ u.T

# confv = get_correctors_strength()
confv = sofb.kickch * 1e-6

errors = imat @ confv

corrs = []
relation = []
c2 = np.dot(confv, confv)
for i in range(mat.shape[1]):
    m = mat[:, i]
    corrs.append(np.dot(confv, m) / np.dot(m, m))
    relation.append(np.dot(confv, m)**2 / np.dot(m, m) / c2)
for fam, err, ang, rel in zip(fams, errors, angs, relation):
    print('{:5s}: {:-6.1f} {:6.1f} {:4.1f}'.format(
        fam, err*1e6, -err/ang*100,  rel*100))
fit = mat @ errors

f = plt.figure(figsize=(8, 5))
gs = mpl_gs.GridSpec(1, 1)
gs.update(
    left=0.10, right=0.97, top=0.97, bottom=0.05, hspace=0.4, wspace=0.25)
ax = plt.subplot(gs[0, 0])
ax.grid(True)
ax.set_ylabel(r'CH Kick [$\mu$rad]')
ax.set_xlabel('CH Index')

print('std real', np.std(confv)*1e6)
print('std diff', np.std(confv - fit)*1e6)

ax.plot(1e6*confv, '-o', label='Before Correction')
# ax.plot(1e6*fit, '-o', label='Fitted with B1B2')
ax.plot(1e6*(confv - fit), '-o', label='Predicted After Correction')
ax.plot(sofb.kickch, '-o', label='After Correction')
ax.legend()

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

plt.show()
