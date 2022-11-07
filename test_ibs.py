############################################################
#################IBS as function of current#################
############################################################
import time
import numpy as np
import pyaccel as pa
from pymodels import si
import matplotlib.pyplot as mplt

mplt.rcParams.update({
    'font.size': 18, 'axes.grid': True, 'lines.linewidth': 2, 
    'grid.alpha': 0.5, 'grid.linestyle': '--'})

mod = si.create_accelerator()
famd = si.get_family_data(mod)

mod[famd['QS']['index'][0][0]].KsL = 0.01
ibs = pa.intrabeam_scattering.IBS(mod)
ibs.curr_per_bunch = 1e-3
ibs.delta_time = 1/10
ibs.relative_tolerance = 1e-4
print(ibs)

npts = 40
current = np.logspace(-2, np.log10(3), npts) * 1e-3
emit1 = np.zeros((npts, 4))
emit2 = np.zeros((npts, 4))
espread = np.zeros((npts, 4))
bunlen = np.zeros((npts, 4))
for i, curr in enumerate(current):
    t0 = time.time()
    ibs.curr_per_bunch = curr
    emit1[i, 0] = ibs.emit10
    emit2[i, 0] = ibs.emit20
    espread[i, 0] = ibs.espread0
    bunlen[i, 0] = ibs.bunlen0

    ibs.ibs_model = ibs.IBS_MODEL.CIMP
    ibs.calc_ibs()
    emit1[i, 1] = ibs.emit1
    emit2[i, 1] = ibs.emit2
    espread[i, 1] = ibs.espread
    bunlen[i, 1] = ibs.bunlen

    ibs.ibs_model = ibs.IBS_MODEL.Bane
    ibs.calc_ibs()
    emit1[i, 2] = ibs.emit1
    emit2[i, 2] = ibs.emit2
    espread[i, 2] = ibs.espread
    bunlen[i, 2] = ibs.bunlen

    ibs.ibs_model = ibs.IBS_MODEL.BM
    ibs.calc_ibs()
    emit1[i, 3] = ibs.emit1
    emit2[i, 3] = ibs.emit2
    espread[i, 3] = ibs.espread
    bunlen[i, 3] = ibs.bunlen
    print(
        f'{i+1:03d}/{current.size:03d}: {curr*1e3:9.4f} mA'
        f'  (ET: {time.time()-t0:.1f}s)')

fig, ((ae1, ae2), (aes, abl)) = mplt.subplots(
    2, 2, figsize=(10, 6), sharex=True)

title = f'IBS @ e2/e1={ibs.emit20/ibs.emit10*100:.2f} %'
fig.suptitle(title)

ae1.plot(current*1e3, emit1*1e12)
ae2.plot(current*1e3, emit2*1e12)
aes.plot(current*1e3, espread*1e2)
abl.plot(current*1e3, bunlen*1e3)

ae1.set_xscale('log')
aes.set_xlabel('Current per Bunch [mA]')
abl.set_xlabel('Current per Bunch [mA]')

ae1.set_ylabel(r'$\varepsilon_1$ [pm.rad]')
ae2.set_ylabel(r'$\varepsilon_2$ [pm.rad]')
aes.set_ylabel(r'$\sigma_\delta$ [%]')
abl.set_ylabel(r'$\sigma_L$ [mm]')

ae1.legend(['No IBS', 'CIMP', 'Bane', 'BM'], loc='best', fontsize='x-small')

fig.tight_layout()
fig.show()


############################################################
#################IBS as function of coupling################
############################################################
import time
import numpy as np
import pyaccel as pa
from pymodels import si
import matplotlib.pyplot as mplt

mplt.rcParams.update({
    'font.size': 18, 'axes.grid': True, 'lines.linewidth': 2, 
    'grid.alpha': 0.5, 'grid.linestyle': '--'})

mod = si.create_accelerator()
famd = si.get_family_data(mod)

mod[famd['QS']['index'][0][0]].KsL = 0.0001
ibs = pa.intrabeam_scattering.IBS(mod)
ibs.curr_per_bunch = 1e-3
ibs.delta_time = 1/10
ibs.relative_tolerance = 1e-4
print(ibs)

npts = 40
ksl = np.logspace(-4, -2, npts)
emit_ratio = np.zeros(npts)
emit1 = np.zeros((npts, 4))
emit2 = np.zeros((npts, 4))
espread = np.zeros((npts, 4))
bunlen = np.zeros((npts, 4))
for i, kl in enumerate(ksl):
    t0 = time.time()
    mod[famd['QS']['index'][0][0]].KsL = kl
    ibs.accelerator = mod

    emit_ratio[i] = ibs.emit20/ibs.emit10
    emit1[i, 0] = ibs.emit10
    emit2[i, 0] = ibs.emit20
    espread[i, 0] = ibs.espread0
    bunlen[i, 0] = ibs.bunlen0

    ibs.ibs_model = ibs.IBS_MODEL.CIMP
    ibs.calc_ibs()
    emit1[i, 1] = ibs.emit1
    emit2[i, 1] = ibs.emit2
    espread[i, 1] = ibs.espread
    bunlen[i, 1] = ibs.bunlen

    ibs.ibs_model = ibs.IBS_MODEL.Bane
    ibs.calc_ibs()
    emit1[i, 2] = ibs.emit1
    emit2[i, 2] = ibs.emit2
    espread[i, 2] = ibs.espread
    bunlen[i, 2] = ibs.bunlen

    ibs.ibs_model = ibs.IBS_MODEL.BM
    ibs.calc_ibs()
    emit1[i, 3] = ibs.emit1
    emit2[i, 3] = ibs.emit2
    espread[i, 3] = ibs.espread
    bunlen[i, 3] = ibs.bunlen
    print(
        f'{i+1:03d}/{ksl.size:03d}: KsL = {kl*1e4:7.2f} G'
        f'  (ET: {time.time()-t0:.1f}s)')

fig, ((ae1, ae2), (aes, abl)) = mplt.subplots(
    2, 2, figsize=(10, 6), sharex=True)

title = f'IBS @ Ib={ibs.curr_per_bunch*1e3:.2f} mA'
fig.suptitle(title)

ae1.plot(emit_ratio*100, emit1*1e12)
ae2.plot(emit_ratio*100, emit2*1e12)
aes.plot(emit_ratio*100, espread*1e2)
abl.plot(emit_ratio*100, 10*bunlen*np.sqrt(emit1*emit2)*1e6**3)

ae1.set_xscale('log')
aes.set_xlabel(r'$\varepsilon_{2,0}/\varepsilon_{1,0}$ [%]')
abl.set_xlabel(r'$\varepsilon_{2,0}/\varepsilon_{1,0}$ [%]')

ae1.set_ylabel(r'$\varepsilon_1$ [pm.rad]')
ae2.set_ylabel(r'$\varepsilon_2$ [pm.rad]')
aes.set_ylabel(r'$\sigma_\delta$ [%]')
abl.set_ylabel(r'Volume [$\mu m^3$]')

ae1.legend(
    ['No IBS', 'CIMP', 'Bane', 'BM'],
    loc='best', fontsize='x-small')

ae1.set_yscale('log')
ae2.set_yscale('log')
aes.set_yscale('log')
abl.set_yscale('log')

fig.tight_layout()
fig.show()


############################################################
##################IBS as function of time###################
############################################################
import numpy as np
import pyaccel as pa
from pymodels import si
import matplotlib.pyplot as mplt

mplt.rcParams.update({
    'font.size': 18, 'axes.grid': True, 'lines.linewidth': 2, 
    'grid.alpha': 0.5, 'grid.linestyle': '--'})

mod = si.create_accelerator()
famd = si.get_family_data(mod)

mod[famd['QS']['index'][0][0]].KsL = 0.01
ibs = pa.intrabeam_scattering.IBS(mod)
ibs.curr_per_bunch = 1e-3
ibs.delta_time = 1/10
ibs.relative_tolerance = 1e-4
print(ibs)

mod[famd['QS']['index'][0][0]].KsL = 0.001
ibs = pa.intrabeam_scattering.IBS(mod)
ibs.ibs_model = ibs.IBS_MODEL.CIMP
data_cimp = ibs.calc_ibs()
ibs.ibs_model = ibs.IBS_MODEL.Bane
data_bane = ibs.calc_ibs()
ibs.ibs_model = ibs.IBS_MODEL.BM
data_bjmt = ibs.calc_ibs()


fig = mplt.figure(figsize=(10, 10))
gs = mplt.GridSpec(
    2, 1, height_ratios=[2.5, 1], top=0.95, bottom=0.07,
    hspace=0.25, right=0.98)
gsr = gs[0, 0].subgridspec(3, 1, hspace=0.03)
gse = gs[1, 0].subgridspec(1, 3, wspace=0.5)
ar1 = mplt.subplot(gsr[0, 0])
ar2 = mplt.subplot(gsr[1, 0], sharex=ar1)
ar3 = mplt.subplot(gsr[2, 0], sharex=ar1)
ae1 = mplt.subplot(gse[0, 0])
ae2 = mplt.subplot(gse[0, 1], sharex=ae1)
aes = mplt.subplot(gse[0, 2], sharex=ae1)

title = (
    f'IBS @ Ib={ibs.curr_per_bunch*1e3:.2f} mA, '
    f'e2/e1={ibs.emit20/ibs.emit10*100:.2f} %')
ar1.set_title(title)
ar1.plot(data_cimp['spos'], data_cimp['growth_rates'][0][0], color='C0', label='X')
ar1.plot(data_bane['spos'], data_bane['growth_rates'][0][0], color='C1')
ar1.plot(data_bjmt['spos'], data_bjmt['growth_rates'][0][0], color='C2')

ar2.plot(data_cimp['spos'], data_cimp['growth_rates'][0][1], color='C0', label='Y')
ar2.plot(data_bane['spos'], data_bane['growth_rates'][0][1], color='C1')
ar2.plot(data_bjmt['spos'], data_bjmt['growth_rates'][0][1], color='C2')

ar3.plot(data_cimp['spos'], data_cimp['growth_rates'][0][2], color='C0', label='L')
ar3.plot(data_bane['spos'], data_bane['growth_rates'][0][2], color='C1')
ar3.plot(data_bjmt['spos'], data_bjmt['growth_rates'][0][2], color='C2')
pa.graphics.draw_lattice(mod, height=20, gca=ar3)

ar1.set_ylabel(r'$\alpha^\mathrm{IBS}_{1,0}$ [1/s]')
ar2.set_ylabel(r'$\alpha^\mathrm{IBS}_{2,0}$ [1/s]')
ar3.set_ylabel(r'$\alpha^\mathrm{IBS}_{3,0}$ [1/s]')
ar3.set_xlabel('Position [m]')
ar1.set_xlim([0, mod.length/20])

ar1.legend(['CIMP', 'Bane', 'BM'], loc='best', fontsize='x-small')

ae1.plot(data_cimp['tim']*1e3, data_cimp['emit1']*1e12)
ae1.plot(data_bane['tim']*1e3, data_bane['emit1']*1e12)
ae1.plot(data_bjmt['tim']*1e3, data_bjmt['emit1']*1e12)
ae2.plot(data_cimp['tim']*1e3, data_cimp['emit2']*1e15)
ae2.plot(data_bane['tim']*1e3, data_bane['emit2']*1e15)
ae2.plot(data_bjmt['tim']*1e3, data_bjmt['emit2']*1e15)
aes.plot(data_cimp['tim']*1e3, data_cimp['espread']*1e4)
aes.plot(data_bane['tim']*1e3, data_bane['espread']*1e4)
aes.plot(data_bjmt['tim']*1e3, data_bjmt['espread']*1e4)

ae1.set_ylabel(r'$\varepsilon_1$ [pm.rad]')
ae2.set_ylabel(r'$\varepsilon_2$ [fm.rad]')
aes.set_ylabel(r'$\sigma_\delta \times 10^4$')
ae1.set_xlabel('Time [ms]')
ae2.set_xlabel('Time [ms]')
aes.set_xlabel('Time [ms]')

ae1.legend(['CIMP', 'Bane', 'BM'], loc='best', fontsize='x-small')

fig.show()

############################################################
#################IBS as function of time step###############
############################################################


import pyaccel as pa
from pymodels import si
import matplotlib.pyplot as mplt
import numpy as np

mplt.rcParams.update({
    'font.size': 18, 'axes.grid': True, 'lines.linewidth': 2, 
    'grid.alpha': 0.5, 'grid.linestyle': '--'})

mod = si.create_accelerator()
famd = si.get_family_data(mod)

mod[famd['QS']['index'][0][0]].KsL = 0.01
ibs = pa.intrabeam_scattering.IBS(mod)
ibs.curr_per_bunch = 1e-3
ibs.delta_time = 1/10
ibs.relative_tolerance = 1e-4
ibs.ibs_model = ibs.IBS_MODEL.BM
print(ibs)

npts = 40
ksl = np.linspace(1e-5, 1e-2, npts)
emit_ratio = np.zeros(npts)
emit1 = np.zeros((npts, 4))
emit2 = np.zeros((npts, 4))
espread = np.zeros((npts, 4))
bunlen = np.zeros((npts, 4))
for i, kl in enumerate(ksl):
    mod[famd['QS']['index'][0][0]].KsL = kl
    ibs.accelerator = mod
    emit_ratio[i] = ibs.emit20/ibs.emit10
    emit1[i, 0] = ibs.emit10
    emit2[i, 0] = ibs.emit20
    espread[i, 0] = ibs.espread0
    bunlen[i, 0] = ibs.bunlen0

    ibs.delta_time = 1/10
    ibs.calc_ibs()
    emit1[i, 1] = ibs.emit1
    emit2[i, 1] = ibs.emit2
    espread[i, 1] = ibs.espread
    bunlen[i, 1] = ibs.bunlen

    ibs.delta_time = 1/5
    ibs.calc_ibs()
    emit1[i, 2] = ibs.emit1
    emit2[i, 2] = ibs.emit2
    espread[i, 2] = ibs.espread
    bunlen[i, 2] = ibs.bunlen

    ibs.delta_time = 1/1
    ibs.calc_ibs()
    emit1[i, 3] = ibs.emit1
    emit2[i, 3] = ibs.emit2
    espread[i, 3] = ibs.espread
    bunlen[i, 3] = ibs.bunlen
    print(f'{i+1:03d}/{ksl.size:03d}: {kl*1e4:.2f} G')

fig, ((ae1, ae2), (aes, abl)) = mplt.subplots(
    2, 2, figsize=(14, 10), sharex=True)

title = f'IBS with BM Model @ Ib={ibs.curr_per_bunch*1e3:.2f} mA'
fig.suptitle(title)
ae1.plot(emit_ratio*100, emit1*1e12)
ae2.plot(emit_ratio*100, emit2*1e12)
aes.plot(emit_ratio*100, espread*1e2)
abl.plot(emit_ratio*100, bunlen*np.sqrt(emit1*emit2)*1e6**3)

ae1.set_xscale('log')
aes.set_xlabel('Emittance Ratio [%]')
abl.set_xlabel('Emittance Ratio [%]')

ae1.set_ylabel('Emmitance 1 [pm.rad]')
ae2.set_ylabel('Emmitance 2 [pm.rad]')
aes.set_ylabel('Energy Spread [%]')
abl.set_ylabel('Volume [um^3]')

ae1.legend(['w/IBS', '1/10', '1/5', '1'], loc='best', fontsize='x-small')

fig.tight_layout()
fig.show()

