import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
from matplotlib import rcParams

from pymodels import si
import pyaccel

rcParams.update(
    {'font.size': 16, 'lines.linewidth': 2, 'axes.grid': True,
    'grid.alpha': 0.5, 'grid.linestyle': '--', 'lines.markersize': 10})


def find_params(mod, twi, indcs, famname, extang=6, header=False):
    if header:
        print(f'{"FAM":4s} {"Pos":^5s} --> | {"bx":^5s} || {"by":^5s} || {"ax":^5s} || {"ay":^5s} || {"gx":^5s} || {"gy":^5s} || {"ex":^5s} || {"elx":^5s} || {"sxb":^5s} || {"sx":^5s} || {"sy":^5s} || {"dxb":^5s} || {"dx":^5s} || {"dy":^5s}')
        return
    ang = 0
    idc = None
    extang *= 1e-3
    for ind in indcs:
        ang += mod[ind].angle
        if ang >= extang:
            idc = ind
            break
    bx = twi[idc].betax
    by = twi[idc].betay
    ax = twi[idc].alphax
    ay = twi[idc].alphay
    ex = twi[idc].etax
    ey = twi[idc].etay
    elx = twi[idc].etapx
    ely = twi[idc].etapy
    gx = (1 + ax*ax)/bx
    gy = (1 + ay*ay)/by
    emt0 = 251e-12
    sprd = 0.9e-3
    k = 3e-2
    emtx = emt0 / (1 + k)
    emty = emt0 * k / (1 + k)
    sxb = (bx * emtx)**(1/2)
    syb = (by * emty)**(1/2)
    sx = (sxb*sxb + ex*sprd*ex*sprd)**(1/2)
    sy = (syb*syb + ey*sprd*ey*sprd)**(1/2)
    dxb = (gx*emtx)**(1/2)
    dyb = (gy*emty)**(1/2)
    dx = (dxb*dxb + elx*sprd*elx*sprd)**(1/2)
    dy = (dyb*dyb + ely*sprd*ely*sprd)**(1/2)
    print(f'{famname:4s} {twi[idc].spos:^5.1f} --> | {bx:^5.1f} || {by:^5.1f} || {ax:^5.1f} || {ay:^5.1f} || {gx:^5.1f} || {gy:^5.1f} || {ex*1e3:^5.1f} || {elx*1e3:^5.1f} || {sxb*1e6:^5.1f} || {sx*1e6:^5.1f} || {sy*1e6:^5.1f} || {dxb*1e6:^5.1f} || {dx*1e6:^5.1f} || {dy*1e6:^5.1f}')
    return idc


mod = si.create_accelerator()
idcs = []
for i, el in enumerate(mod):
    if not el.pass_method.startswith('identity_pass'):
        idcs.append(i)
twi, *_ = pyaccel.optics.calc_twiss(mod)

flds = [0, ] * len(mod)
for i, el in enumerate(mod):
    if el.length > 0:
        flds[i] = el.angle / el.length
flds = np.array(flds)*10

fam_names = ['B1', 'B2', 'mc', 'mia', 'mib']
extangs = [6, 20, 0, 0, 0]
symbs = ['o', '*', 'd', 's', 'p']
inds = [pyaccel.lattice.find_indices(mod, 'fam_name', f) for f in fam_names]
find_params(0, 0, 0, 0, 0, header=True)
ind = [
    find_params(mod, twi, i, f, extang=a) for i, f, a in
        zip(inds, fam_names, extangs)]

fig = plt.figure(figsize=(9, 9))
gs = mgs.GridSpec(2, 1)
gs.update(left=0.1, bottom=0.09, top=0.94, right=0.92, hspace=0.1)

fig.suptitle('Diagnostic Beamlines Extraction Points')

ax1 = fig.add_subplot(gs[0, 0])
ax2 = ax1.twinx()
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

ax1.plot(twi.spos[idcs], twi.betax[idcs], color='tab:blue', label=r'$\beta_x$')
ax1.plot(twi.spos, twi.betay, color='tab:red', label=r'$\beta_y$')
ax2.plot(twi.spos, twi.etax*100, color='tab:green')
for i, s in zip(ind, symbs):
    ax1.plot(twi[i].spos, twi[i].betax, s, color='tab:blue')
    ax1.plot(twi[i].spos, twi[i].betay, s, color='tab:red')
    ax2.plot(twi[i].spos, twi[i].etax*100, s, color='tab:green')
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.xaxis.label.set_visible(False)

ax1.legend(loc='best')
ax1.set_ylabel(r'$\beta$ [m]')
ax1.set_xlabel(r'Position [m]')
ax2.set_ylabel(r'$\eta_x$ [cm]')

ax2.tick_params(axis='y', colors='tab:green')
ax2.spines['right'].set_color('tab:green')
ax2.yaxis.label.set_color('tab:green')

ax3.plot(twi.spos[idcs], flds[idcs], color='black')
for i, s, f in zip(ind, symbs, fam_names):
    ax3.plot(twi[i].spos, flds[i], s, color='black', label=f)
ax3.set_xlabel('Position [m]')
ax3.set_ylabel('Dipole Field [T]')
ax3.legend(loc='best')

ax3.set_ylim([0, 0.7])
ax1.set_xlim([0, 518.4/20])

plt.show()
