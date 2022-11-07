import time

import numpy as np
import scipy.linalg as scylin

import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs
from pymodels import si
import pyaccel as pyacc

np.set_printoptions(precision=4, linewidth=130)

J = np.kron(np.eye(3, dtype=int), np.array([[0, 1], [-1, 0]]))


def test_ohmienvelope():
    energy_offset = -0/1000
    model = si.create_accelerator()
    model.radiation_on = True
    model.cavity_on = True

    famdata = si.get_family_data(model)

    model[famdata['QS']['index'][0][0]].KsL = 0.0

    twi, *_ = pyacc.optics.calc_twiss(
        model, energy_offset=energy_offset, indices='closed')

    tini = time.time()
    # envs = calc_ohmienvelope(model)
    envs, cummat, bdiff = pyacc.optics.calc_ohmienvelope(
        model, energy_offset=energy_offset, full=True, indices='closed')
    print(time.time()-tini)

    m66 = cummat[-1]

    m66i = cummat @ m66 @ np.linalg.inv(cummat)

    evals, evecs = np.linalg.eig(m66i)
    evecsi = np.linalg.inv(evecs)
    evecsih = evecsi.swapaxes(-1, -2).conj()

    eval0 = evals[0]
    trc = (eval0[::2] + eval0[1::2]).real
    dff = (eval0[::2] - eval0[1::2]).imag
    mus = np.arctan2(dff, trc)
    alphas = trc / np.cos(mus) / 2
    alphas = -np.log(alphas) * 499664000/864
    print('dampings: ', 1/alphas*1e3)
    print('tunes: ', mus/2/np.pi)

    env0r = evecsi @ envs @ evecsih

    emits = np.diagonal(env0r, axis1=-1, axis2=-2).real[:, ::2] * 1e12
    emits /= np.linalg.norm(evecsi, axis=-1)[:, ::2]
    emits = np.sort(emits)

    spread = np.sqrt(envs[0, 4, 4])
    bunlen = np.sqrt(envs[0, 5, 5])
    print(spread, bunlen)

    # mplt.semilogy(twi.spos, emits[:-1, :])
    # # mplt.semilogy(twi.spos, emitx[:-1])
    # mplt.ylabel('emits [$\mu$m.$\mu$rad]')
    # mplt.xlabel('index')
    # mplt.grid(True, alpha=0.5, linestyle='--')

    # sigmax = np.sqrt(emits[:, 1]/1e12*twi.betax + (spread*twi.etax)**2)
    # sigmx = np.sqrt(envs[:, 0, 0])
    # # sigmx = envs[:, 0, 0]
    # mplt.plot(twi.spos, sigmax, label='Twiss')
    # mplt.plot(twi.spos, sigmx, label='Ohmi')
    # mplt.plot(twi.spos, (sigmax - sigmx)*1000, label='Diff x 1000')
    # mplt.grid(True, alpha=0.5, linestyle='--')

    # mplt.legend()
    # mplt.tight_layout()
    # mplt.show()


def test_energy_offset():
    energy_offset = 20/1000
    model = si.create_accelerator()
    model.radiation_on = True
    model.cavity_on = True

    ind = pyacc.lattice.find_indices(
        model, 'frequency', 200, comparison=lambda x, y: x > y)
    ind = ind[0]
    model[ind].frequency *= 1 - energy_offset * 1.697e-4

    passm = [ps for ps in pyacc.elements.PASS_METHODS if 'sympl' in ps]
    model2 = pyacc.lattice.refine_lattice(
        model, max_length=0.02, pass_methods=passm)

    eqint = pyacc.optics.EquilibriumParametersIntegrals(
        model2, energy_offset=energy_offset)

    spos = pyacc.lattice.find_spos(model2, indices='closed')
    eqohmi = pyacc.optics.EquilibriumParametersOhmiFormalism(
        model2, energy_offset=energy_offset)

    print(eqint)
    print(eqohmi)

    mplt.plot(
        eqint.twiss.spos, eqint.sigma_rx*1e6,
        label=f'Integrals ($\epsilon_x$ = {eqint.emitx*1e12:.2f} pm.rad, $\sigma_\delta$ = {eqint.espread0*1e4:.2f})')
    mplt.plot(
        spos, eqohmi.sigma_rx*1e6,
        label=f'Ohmi ($\epsilon_x$ = {eqohmi.emitx*1e12:.2f} pm.rad, $\sigma_\delta$ = {eqohmi.espread0*1e4:.2f})')
    mplt.grid(True, alpha=0.5, linestyle='--')

    mplt.legend()
    mplt.tight_layout()
    mplt.show()


def energy_offset_dependency():
    model = si.create_accelerator()
    model.radiation_on = True
    model.cavity_on = True

    energy_offset = np.linspace(-3, 3, 30) / 100

    ind = pyacc.lattice.find_indices(
        model, 'frequency', 200, comparison=lambda x, y: x > y)
    ind = ind[0]
    freq0 = model[ind].frequency

    emit, spread = [], []
    for eoff in energy_offset:
        eoff = float(eoff)
        model[ind].frequency = freq0*(1 - eoff*1.697e-4)
        eqohmi = pyacc.optics.EquilibriumParametersOhmiFormalism(
            model, energy_offset=eoff)
        emit.append(eqohmi.emitx)
        spread.append(eqohmi.espread0)

    emit = np.array(emit) * 1e12
    spread = np.array(spread) * 1e4

    fig = mplt.figure(figsize=(6, 4))
    gs = mgs.GridSpec(1, 1, left=0.12, right=0.88, bottom=0.12, top=0.98)
    ax = fig.add_subplot(gs[0, 0])
    ay = ax.twinx()

    ax.plot(energy_offset*100, emit, color='C0')
    ay.plot(energy_offset*100, spread, color='C1')
    ax.grid(True, alpha=0.5, linestyle='--')
    mplt.setp(ax.get_yticklabels(), color='C0')
    mplt.setp(ay.get_yticklabels(), color='C1')
    ax.set_ylabel('Horizontal Emittance [pm.rad]', color='C0')
    ay.set_ylabel('Energy Spread x 10000', color='C1')
    ax.set_xlabel('Energy Deviation [%]')
    return fig


def ksl_dependency():
    model = si.create_accelerator()
    model.radiation_on = True
    model.cavity_on = True

    ksl = np.linspace(0, 6, 31) / 100

    famdata = si.get_family_data(model)
    ind = famdata['QS']['index'][0][0]

    emitx, emity = [], []
    for eoff in ksl:
        eoff = float(eoff)
        model[ind].KsL = eoff
        eqohmi = pyacc.optics.EquilibriumParametersOhmiFormalism(model)
        emitx.append(eqohmi.emitx)
        emity.append(eqohmi.emity)

    emitx = np.array(emitx) * 1e12
    emity = np.array(emity) * 1e12

    fig = mplt.figure(figsize=(6, 4))
    gs = mgs.GridSpec(1, 1, left=0.12, right=0.88, bottom=0.12, top=0.98)
    ax = fig.add_subplot(gs[0, 0])
    # ay = ax.twinx()

    demitx = emitx - emitx[0]
    demity = emity - emity[0]
    ax.plot(emity/emitx*100, demitx, color='C0', label='Horizontal')
    ax.plot(emity/emitx*100, demity, color='C1', label='Vertical')
    ax.plot(emity/emitx*100, demitx + demity, color='C3', label='Total')
    # ay.plot(emity/emitx*100, emity/emitx * 100, color='C2')
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.set_ylabel('Emittance [pm.rad]')
    # ay.set_ylabel('Emittance Ratio [%]', color='C2')
    # mplt.setp(ay.get_yticklabels(), color='C2')
    # ax.set_xlabel('KsL [1/km]')
    ax.set_xlabel('Emittance Ratio [%]')
    ax.legend(loc='best')
    return fig


def test_tunesep():

    model = si.create_accelerator()
    model.radiation_on = True
    model.cavity_on = True

    famdata = si.get_family_data(model)
    ind = famdata['QS']['index'][0][0]
    model[ind].KsL = 0.0055

    idcs = np.array(famdata['QFB']['index'], dtype=int).ravel()

    dkl = np.linspace(-0.1, 0.25, 41)/100
    # dkl = np.array([0.0, ])
    kl0 = model[idcs[0]].KL

    emit1 = np.zeros(dkl.shape)
    emit2 = np.zeros(dkl.shape)
    emitx = np.zeros(dkl.shape)
    emity = np.zeros(dkl.shape)
    tunex = np.zeros(dkl.shape)
    tuney = np.zeros(dkl.shape)
    for idx, eoff in enumerate(dkl):
        eoff = float(eoff)
        pyacc.lattice.set_attribute(model, 'KL', idcs, kl0*(1+eoff))
        eqohmi = pyacc.optics.EquilibriumParametersOhmiFormalism(model)
        emitx[idx] = np.sqrt(np.linalg.det(eqohmi.envelopes[0][:2, :2]))
        emity[idx] = np.sqrt(np.linalg.det(eqohmi.envelopes[0][2:4, 2:4]))
        emit1[idx] = eqohmi.emitx
        emit2[idx] = eqohmi.emity
        tunex[idx] = eqohmi.tunex
        tuney[idx] = eqohmi.tuney

    # print(emitx[0], emity[0])

    fig = mplt.figure(figsize=(9, 6))
    gs = mgs.GridSpec(
        2, 1, left=0.12, right=0.88, bottom=0.12, top=0.98, hspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ay = fig.add_subplot(gs[1, 0])
    az = ay.twinx()

    idx = np.argmin(np.abs(tunex-tuney))
    dnu = np.abs(tunex-tuney)[idx]
    csi = 2*dnu*np.sin(np.pi*(tunex[idx] + tuney[idx]))
    delta = tuney - tunex
    theta = np.arctan2(csi, np.abs(delta))
    ktheo = np.tan(theta/2)**2
    ktheo2 = np.tan(theta)**2 / (np.tan(theta)**2+2)
    ax.plot(dkl*100, tunex, 'o', color='C0', label='Tune x')
    ax.plot(dkl*100, tuney, 'o', color='C1', label='Tune y')
    ay.plot(delta, emitx*1e12, color='C0', label='Emit x')
    ay.plot(delta, emity*1e12, color='C1', label='Emit y')
    # ay.plot(dkl*100, emit1*1e12, color='C2', label='Emit 1')
    # ay.plot(dkl*100, emit2*1e12, color='C3', label='Emit 2')
    # az.plot(dkl*100, emit2/emit1*100, color='C4', label='ratio')
    az.plot(delta, emity/emitx*100, color='C2')
    # az.plot(dkl*100, ktheo*100, color='C6', label='Boaz')
    # az.plot(dkl*100, ktheo2*100, color='C7', label='Guinard')
    ax.text(
        0.5, 0.2,
        r'$\Delta\nu_\mathrm{min} = $ '+'{0:.3f}'.format(dnu),
        horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)
    ax.set_ylabel('Tunes')
    ax.set_xlabel('$\Delta$Kl QFB [%]')
    ay.set_ylabel('Emittances [pm.rad]')
    ay.set_xlabel(r'$\nu_y - \nu_x$')
    az.set_ylabel('Emittance ratio [%]', color='C2')
    az.set_yscale('log')
    mplt.setp(az.get_yticklabels(), color='C2')
    ax.grid(True, alpha=0.5, linestyle='--')
    ay.grid(True, alpha=0.5, linestyle='--')
    mplt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(loc='best')
    ay.legend(loc='best')
    return fig


def main():
    # test_ohmienvelope()
    # test_energy_offset()
    # fig = energy_offset_dependency()
    # fig = ksl_dependency()
    tini = time.time()
    fig = test_tunesep()
    print(time.time()-tini)
    mplt.show()


if __name__ == '__main__':
    main()
