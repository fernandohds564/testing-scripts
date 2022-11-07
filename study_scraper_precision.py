#!/usr/bin/env python-sirius
import os
import time

import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs
from matplotlib import rcParams

from mathphys.functions import save_pickle, load_pickle
from pymodels import si
import pyaccel as pyacc

rcParams.update({
    'axes.grid': True, 'grid.alpha': 0.5, 'grid.linestyle': '--',
    'font.size': 12})


def test_touschek_accep_function(mod):
    """."""
    spos = pyacc.lattice.find_spos(mod, indices='closed')

    kwrgs = dict(
        accelerator=mod, track=False, check_tune=False,
        energy_offsets=np.linspace(0.02, 0.05, 40))

    t0_ = time.time()
    accn0, accp0 = pyacc.optics.calc_touschek_energy_acceptance(**kwrgs)
    t1_ = time.time()
    print(f'Linear took {(t1_-t0_):.3f}s')

    kwrgs['check_tune'] = True
    accn1, accp1 = pyacc.optics.calc_touschek_energy_acceptance(**kwrgs)
    t2_ = time.time()
    print(f'Tune took {(t2_-t1_):.3f}s')

    kwrgs['track'] = True
    accn2, accp2 = pyacc.optics.calc_touschek_energy_acceptance(**kwrgs)
    t3_ = time.time()
    print(f'Tune Track took {(t3_-t2_):.3f}s')

    fig = mplt.figure(figsize=(8, 5))
    gs = mgs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(spos, accp0*100, color='tab:blue', label='Basic')
    ax.plot(spos, accp1*100, color='tab:red', label='Tune')
    ax.plot(spos, accp2*100, color='tab:green', label='Tune & Track')
    ax.plot(spos, accn0*100, color='tab:blue')
    ax.plot(spos, accn1*100, color='tab:red')
    ax.plot(spos, accn2*100, color='tab:green')
    # mplt.sca(ax2)
    pyacc.graphics.draw_lattice(mod, height=1, offset=0, gca=True)
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Enegy Acceptance [%]')
    ax.legend(
        loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize='x-small',
        ncol=4)
    ax.set_xlim([0, 518.4/5])
    fig.tight_layout()
    return fig


def do_vertical_scraper_study(mod):
    """."""
    scrapv = pyacc.lattice.find_indices(mod, 'fam_name', 'ScrapV')
    vmax_orig = pyacc.lattice.get_attribute(mod, 'vmax', scrapv)
    vmin_orig = pyacc.lattice.get_attribute(mod, 'vmin', scrapv)

    aberts = np.linspace(4, 3.9, 30)*1e-3

    twi, *_ = pyacc.optics.calc_twiss(mod)

    ltime = pyacc.lifetime.Lifetime(mod)
    ltime.curr_per_bunch = 500/864

    acc_neg, acc_pos = pyacc.optics.calc_touschek_energy_acceptance(
        mod, track=False)
    rf_accep = ltime.equi_params.rf_acceptance
    acc_neg = np.maximum(acc_neg, -rf_accep)
    acc_pos = np.minimum(acc_pos, rf_accep)
    ltime.accepen = (acc_neg, acc_pos)

    acceps_x = np.zeros(aberts.size)
    acceps_y = np.zeros(aberts.size)
    lossrate_elastic = np.zeros(aberts.size)
    lossrate_inelastic = np.zeros(aberts.size)
    lossrate_touschek = np.zeros(aberts.size)
    lossrate_total = np.zeros(aberts.size)
    for idx, aber in enumerate(aberts):
        pyacc.lattice.set_attribute(mod, 'vmin', scrapv, -aber)
        pyacc.lattice.set_attribute(mod, 'vmax', scrapv, aber)
        res = pyacc.optics.calc_transverse_acceptance(mod, twiss=twi)
        ltime.accepx = res[0].min()
        ltime.accepy = res[1].min()
        acceps_x[idx] = res[0].min()
        acceps_y[idx] = res[1].min()
        lossrate_elastic[idx] = ltime.lossrate_elastic
        lossrate_inelastic[idx] = ltime.lossrate_inelastic
        lossrate_touschek[idx] = ltime.lossrate_touschek
        lossrate_total[idx] = ltime.lossrate_total

    pyacc.lattice.set_attribute(mod, 'vmin', scrapv, vmin_orig)
    pyacc.lattice.set_attribute(mod, 'vmax', scrapv, vmax_orig)

    fig = mplt.figure()
    gs = mgs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ay = ax.twinx()
    ax.plot(aberts*1e3, acceps_y*1e6, '.-')
    ay.plot(
        aberts*1e3, 1/lossrate_elastic/3600, '.-',
        color='tab:red', label='Elastic Lifetime')
    # ay.plot(
    #     aberts*1e3, 1/lossrate_total/3600, '.-',
    #     color='tab:green', label='Total Lifetime')
    mplt.setp(ax.get_yticklabels(), color='tab:blue')
    mplt.setp(ay.get_yticklabels(), color='tab:red')
    ax.set_ylabel('Acceptance [mm.mrad]', color='tab:blue')
    ay.set_ylabel('Elastic Lifetime [h]', color='tab:red')
    ax.set_title(
        'Acceptance and Lifetime @ I={0:.1f}mA P_avg={1:.1f}pbar'.format(
            ltime.curr_per_bunch*864, ltime.avg_pressure*1e9))
    ay.grid(False)
    ax.set_xlabel('ScrapV Aperture [mm]')
    fig.tight_layout()
    return fig


def do_horizontal_scraper_study(mod):
    """."""
    aberts = np.linspace(0.6, 0.1, 51)*1e-3

    twi, *_ = pyacc.optics.calc_twiss(mod, indices='closed')
    spos = twi.spos

    ltime = pyacc.lifetime.Lifetime(mod)
    ltime.curr_per_bunch = 100/864
    ltime.coupling = 0.01

    rf_accep = ltime.equi_params.rf_acceptance

    fname = 'study_scraph_data'
    if not os.path.isfile(fname + '.pickle'):
        scraph = pyacc.lattice.find_indices(mod, 'fam_name', 'ScrapH')
        hmax_orig = pyacc.lattice.get_attribute(mod, 'hmax', scraph)
        hmin_orig = pyacc.lattice.get_attribute(mod, 'hmin', scraph)

        acceps_x = np.zeros(aberts.size)
        acceps_y = np.zeros(aberts.size)
        accepenp = np.zeros((aberts.size, len(twi)))
        accepenn = np.zeros((aberts.size, len(twi)))
        lossrate_elastic = np.zeros(aberts.size)
        lossrate_inelastic = np.zeros(aberts.size)
        lossrate_touschek = np.zeros(aberts.size)
        lossrate_quantumx = np.zeros(aberts.size)
        lossrate_quantumy = np.zeros(aberts.size)
        lossrate_quantume = np.zeros(aberts.size)
        lossrate_quantum = np.zeros(aberts.size)
        lossrate_total = np.zeros(aberts.size)
        energies = np.linspace(0.05, 35, 100) / 1000
        for idx, aber in enumerate(aberts):
            print(f'{aber*1000:.3f} mm')
            pyacc.lattice.set_attribute(mod, 'hmin', scraph, -aber)
            pyacc.lattice.set_attribute(mod, 'hmax', scraph, aber)
            res = pyacc.optics.calc_transverse_acceptance(mod, twiss=twi)
            acc_neg, acc_pos = pyacc.optics.calc_touschek_energy_acceptance(
                mod, track=False, energy_offsets=energies)
            accepenn[idx] = acc_neg
            accepenp[idx] = acc_pos
            acc_neg = np.maximum(acc_neg, -rf_accep)
            acc_pos = np.minimum(acc_pos, rf_accep)
            ltime.accepen = (acc_neg, acc_pos)
            acceps_x[idx] = res[0].min()
            acceps_y[idx] = res[1].min()
            ltime.accepx = acceps_x[idx]
            ltime.accepy = acceps_y[idx]
            lossrate_elastic[idx] = ltime.lossrate_elastic
            lossrate_inelastic[idx] = ltime.lossrate_inelastic
            lossrate_touschek[idx] = ltime.lossrate_touschek
            lossrate_quantumx[idx] = ltime.lossrate_quantumx
            lossrate_quantumy[idx] = ltime.lossrate_quantumy
            lossrate_quantume[idx] = ltime.lossrate_quantume
            lossrate_quantum[idx] = ltime.lossrate_quantum
            lossrate_total[idx] = ltime.lossrate_total

        data = dict(
            aberts=aberts,
            acceps_x=acceps_x,
            acceps_y=acceps_y,
            accepenp=accepenp,
            accepenn=accepenn,
            lossrate_elastic=lossrate_elastic,
            lossrate_inelastic=lossrate_inelastic,
            lossrate_touschek=lossrate_touschek,
            lossrate_quantumx=lossrate_quantumx,
            lossrate_quantumy=lossrate_quantumy,
            lossrate_quantume=lossrate_quantume,
            lossrate_quantum=lossrate_quantum,
            lossrate_total=lossrate_total)
        save_pickle(data, fname)

        pyacc.lattice.set_attribute(mod, 'hmin', scraph, hmin_orig)
        pyacc.lattice.set_attribute(mod, 'hmax', scraph, hmax_orig)

    data = load_pickle(fname)
    aberts = data['aberts'] * 1000
    acceps_x = data['acceps_x']
    acceps_y = data['acceps_y']
    accepenp = data['accepenp']
    accepenn = data['accepenn']
    lossrate_elastic = data['lossrate_elastic']
    lossrate_inelastic = data['lossrate_inelastic']
    lossrate_touschek = data['lossrate_touschek']
    lossrate_quantumx = data['lossrate_quantumx']
    lossrate_quantumy = data['lossrate_quantumy']
    lossrate_quantume = data['lossrate_quantume']
    lossrate_quantum = data['lossrate_quantum']
    lossrate_total = data['lossrate_total']

    lossrate_total = lossrate_elastic + lossrate_inelastic
    lossrate_total += lossrate_touschek + lossrate_quantumy + lossrate_quantumx

    fig = mplt.figure()
    gs = mgs.GridSpec(1, 1)
    ay = fig.add_subplot(gs[0, 0])
    ay.plot(aberts, 1/lossrate_quantumx, '.-', color='C1', label='QuantumX')
    ay.plot(aberts, 1/lossrate_elastic, '.-', color='C2', label='Elastic')
    ay.plot(aberts, 1/lossrate_touschek, '.-', color='C3', label='Touschek')
    ay.plot(aberts, 1/lossrate_total, '.-', color='C4', label='Total')

    ay.set_ylabel('Lifetime [s]')
    ay.set_title('Lifetime @ I={0:.1f}mA P_avg={1:.1f}pbar'.format(
        ltime.curr_per_bunch*864, ltime.avg_pressure*1e9))
    ay.legend(loc='best')
    ay.set_xlabel('ScrapH Aperture [mm]')
    # ay.set_xscale('log')
    ay.set_yscale('log')
    ay.set_ylim([0.1, 1000])
    fig.tight_layout()

    fig = mplt.figure(figsize=(8, 5))
    gs = mgs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    norm = mplt.Normalize(aberts.min(), aberts.max())
    mapp = mplt.cm.ScalarMappable(norm=norm, cmap=mplt.cm.hot)
    for accn, accp, aber in zip(accepenn, accepenp, aberts):
        cor = mapp.to_rgba(aber)
        ax.plot(spos, accp*100, color=cor)
        ax.plot(spos, accn*100, color=cor)

    mplt.colorbar(mapp, ax=ax, label='ScrapH Aperture[mm]')
    pyacc.graphics.draw_lattice(mod, height=1, offset=-4.1, gca=True)
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Enegy Acceptance [%]')
    ax.set_xlim([0, 518.4/5])
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    acc = si.create_accelerator()
    acc.vchamber_on = True
    acc.cavity_on = True
    acc.radiation_on = True

    # fig = do_vertical_scraper_study(acc)
    fig = do_horizontal_scraper_study(acc)
    # fig = test_touschek_accep_function(acc)

    mplt.show()
