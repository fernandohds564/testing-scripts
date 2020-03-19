#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from siriuspy.ramp.ramp import BoosterRamp
from siriuspy.search import MASearch


def find_intersections(t, w):
    d1 = np.diff(w)
    d2 = np.diff(d1)
    ind = np.where(np.abs(d2) < 1e-10)[0]
    dind = np.diff(ind)
    ind2 = np.where(dind > 1)[0]

    idd2 = np.ones(ind2.size+1, dtype=int)*(len(d2)-1)
    idd2[:ind2.size] = ind[ind2]
    idd1 = idd2 + 1
    idw = idd1 + 1

    a = d1[idd1]
    b = w[idw] - a*idw
    ind_inter = -np.diff(b)/np.diff(a)
    w_inter = a[:-1]*ind_inter + b[:-1]

    t_inter = np.interp(ind_inter, np.arange(t.size), t)
    # print(ind_inter)
    # plt.figure()
    # plt.plot(d2)
    # plt.plot(idd2, d2[idd2])
    # plt.figure()
    # plt.plot(d1)
    # plt.plot(idd1, d1[idd1])
    # plt.figure()
    # plt.plot(w, '.-')
    # plt.plot(idw, w[idw], '.-')
    # plt.plot(ind_inter, w_inter, '.-')
    # plt.show()
    return t_inter, w_inter


if __name__ == '__main__':
    ramp = BoosterRamp('mb_bo_Vorbit_change_1')
    ramp.load()

    mas = MASearch.get_manames({'sec': 'BO', 'dis': 'MA'})
    f = plt.figure()
    ax = plt.gca()
    oversamp = 5
    for i, ma in enumerate(mas):
        if ma == ramp.MANAME_DIPOLE:
            continue
        # if ma.sub == 'Fam':
        #     continue
        t = ramp.ps_waveform_get_times(ma)
        w = ramp.ps_waveform_get_strengths(ma)

        oind = np.arange(t.size)
        ind = np.linspace(0, oind[-1], t.size*oversamp)
        t2 = np.interp(ind, oind, t)
        w2 = np.interp(ind, oind, w)
        t_inter, w_inter = find_intersections(t2, w2)
        t_inter2 = np.round(t_inter, decimals=3)
        if not t_inter.size:
            continue
        w_inter2 = np.interp(t_inter2, t_inter, w_inter)
        w_rec = np.interp(t_inter2, t, w)
        vals = ', '.join([
            '{:10.2g}'.format((idx-idx2)) for idx, idx2 in zip(w_inter2, w_rec)
            ])
        print('{0:12s} --> {1:s}'.format(ma, vals))
        ax.plot(t, w)
        ax.plot(t_inter, w_inter)

    plt.show()
