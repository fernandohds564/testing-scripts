#!/usr/bin/env python-sirius

import time

import numpy as np
from scipy.optimize import curve_fit
from epics import PV, ca

import matplotlib.pyplot as plt
import matplotlib.cm as mcmap
import matplotlib.gridspec as mgrid
from matplotlib import rcParams

from test_ioc import PVTMPL

rcParams.update({
    'font.size': 16, 'lines.linewidth': 2, 'axes.grid': True,
    'text.usetex': True})


def _set_pvs(pvs, values):
    """."""
    for pv, value in zip(pvs, values):
        pv.put(value, wait=False)


def _compare_pvs(pvs, values):
    """."""
    while True:
        vals = []
        for pv in pvs:
            ca.get(pv.chid, wait=False)
        for pv in pvs:
            val = ca.get_complete(pv.chid)
            if val is None:
                raise ValueError('val is None')
            vals.append(val)
        vals = np.asarray(vals, dtype=float)
        eq = np.allclose(vals, values, atol=1e-7)
        if eq:
            break
        # else:
        #     print(np.max(vals - values))
        time.sleep(0.001)
        # for pvo, value in zip(pvs, values):
        #     if not pvo.connected:
        #         time.sleep(0.001)
        #         break
        #     val = pvo.value
        #     if val is None:
        #         time.sleep(0.001)
        #         break
        #     vec1 = np.asarray(val, dtype=np.float32)
        #     vec2 = np.asarray(value, dtype=np.float32)
        #     if not np.all(vec1 == vec2):
        #         time.sleep(0.001)
        #         break
        # else:
        #     break


# #########################################################
# ################ Test IOC Performance ###################
# #########################################################
def test_performance():
    """."""
    pvssp = []
    pvsrb = []
    for pref in range(40):
        for pvi in range(1):
            pvssp.append(PV(PVTMPL(pref, pvi, 'SP'), auto_monitor=False))
            pvsrb.append(PV(PVTMPL(pref, pvi, 'RB'), auto_monitor=False))

    for pvs, pvr in zip(pvssp, pvsrb):
        if not pvs.wait_for_connection():
            raise ValueError('not connected')
        if not pvr.wait_for_connection():
            raise ValueError('not connected')

    time.sleep(1)
    times_sp, times_rb = [], []
    for i in range(1, 501):
        values = np.random.rand(len(pvssp), 8)
        values = np.asarray(values)
        t0 = time.time()
        _set_pvs(pvssp, values)
        t1 = time.time()
        _compare_pvs(pvsrb, values)
        t2 = time.time()
        times_sp.append(t1-t0)
        times_rb.append(t2-t1)
        print('.', end='\n' if not i % 50 else '')
        time.sleep(1/100)
    times = np.array([times_sp, times_rb])
    print()

    np.savetxt('ioc_performance.txt', times)


# #########################################################
# ############## Analyse IOC Performance ##################
# #########################################################
def analysis_performance():
    """."""
    def fit_gauss(binc, den, A, x0, sig):
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma*sigma))

        coeff, var_matrix = curve_fit(
            gauss, binc, den, p0=[A, x0, sig])

        # Get the fitted curve
        return gauss(binc, *coeff), coeff

    tstamp = np.loadtxt('ioc_performance.txt').T
    dtimes = tstamp * 1000
    dt_avg = dtimes.mean(axis=1)
    dt_std = dtimes.std(axis=1)
    dt_min = dtimes.min(axis=1)
    dt_max = dtimes.max(axis=1)
    dt_p2p = dt_max - dt_min

    fig = plt.figure(figsize=(10, 10))
    gs = mgrid.GridSpec(3, 1, figure=fig)
    gs.update(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.45)
    ax = plt.subplot(gs[0, 0])
    ay = plt.subplot(gs[1, 0])
    az = plt.subplot(gs[2, 0])

    ax.plot(dtimes, 'o-')
    ax.set_xlabel('aquisition number')
    ax.set_ylabel('dtime [ms]')
    ax.legend(['Set', 'Update'])

    vals = np.array([dt_std, dt_min, dt_avg, dt_p2p, dt_max]).T
    x = np.arange(vals.shape[1])
    ay.bar(x, vals[0])
    ay.bar(x, vals[1])
    ay.set_xticklabels(('', 'STD', 'MIN', 'AVG', 'P2P', 'MAX'))
    # ay.set_xlabel('Stats')
    ay.set_ylabel('dtime [ms]')

    dens, bins, _ = az.hist(dtimes, bins=100, stacked=True)
    az.set_xlabel('dtime [ms]')
    az.set_ylabel('number of occurencies')

    # bins += (bins[1]-bins[0])/2
    # bins = bins[:-1]
    # lower = (bins > 25) & (bins < 34.5)
    # dens1 = dens[lower]
    # bins1 = bins[lower]
    # fitb1, coeff1 = fit_gauss(
    #     bins1, dens1, dens1.max(), bins1.mean(), dens1.std())
    # az.plot(bins1, fitb1, 'tab:orange')
    # txt = r'$\mu$ = '+f'{coeff1[1]:.1f} ms\n'
    # txt += r'$\sigma$ = '+f'{coeff1[2]:.1f} ms'
    # az.annotate(
    #     txt, xy=(coeff1[1], coeff1[0]),
    #     xytext=(coeff1[1] - 3*coeff1[2], coeff1[0]*0.6),
    #     horizontalalignment='right',
    #     arrowprops=dict(facecolor='black', shrink=0.05))

    # upper = bins > 34.5
    # dens2 = dens[upper]
    # bins2 = bins[upper]
    # if dens2.max() > 4:
    #     fitb2, coeff2 = fit_gauss(
    #         bins2, dens2, dens2.max(), bins2.mean(), dens2.std())
    #     az.plot(bins2, fitb2, 'tab:orange')
    #     txt = r'$\mu$ = '+f'{coeff2[1]:.1f} ms\n'
    #     txt += r'$\sigma$ = '+f'{coeff2[2]:.1f} ms'
    #     az.annotate(
    #         txt, xy=(coeff2[1], coeff2[0]),
    #         xytext=(coeff2[1] + 3*coeff2[2], coeff2[0]),
    #         arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()


if __name__ == '__main__':
    test_performance()
    analysis_performance()
