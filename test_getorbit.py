#!/usr/bin/env python-sirius

import time
import datetime
from multiprocessing import Pipe
import pickle

import psutil
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as mcmap
import matplotlib.gridspec as mgrid
from matplotlib import rcParams

from epics import ca, PV, CAProcess
from siriuspy.namesys import SiriusPVName
from siriuspy.sofb.csdev import SOFBFactory
from siriuspy.sofb.orbit import EpicsOrbit

rcParams.update({
    'font.size': 16, 'lines.linewidth': 2, 'axes.grid': True,
    'text.usetex':True})


def save_pickle(fname, data):
    """."""
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'wb') as fil:
        pickle.dump(data, fil)


def load_pickle(fname):
    """."""
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'rb') as fil:
        data = pickle.load(fil)
    return data


class MyFormatter(mticker.Formatter):
    """."""

    def __init__(self, pvs):
        self.pvs_short = [
            f"{pv.sub}{('-'+pv.idx) if pv.idx else ''}" for pv in pvs]

    def __call__(self, x, pos=None):
        x = max(0, min(len(self.pvs_short)-1, int(x)))
        return self.pvs_short[x]


# #########################################################
# ############## synchronous get optimized ################
# #########################################################
def run_synchronous_get_optimized(pvs):
    """Run synchronous optimized get test."""
    pvschid = []
    for pvn in pvs:
        chid = ca.create_channel(pvn, connect=False, auto_cb=False)
        pvschid.append(chid)

    for chid in pvschid:
        ca.connect_channel(chid)

    ca.poll()
    for i in range(600):
        t0 = time.time()
        for chid in pvschid:
            ca.get(chid, wait=False)
        out = []
        for chid in pvschid:
            out.append(ca.get_complete(chid))
        print(f'dtime {(time.time()-t0)*1000:.1f}ms   pos {out[0]:.0f}nm')


# #########################################################
# ###### multiprocess asynchronous get (monitoring) #######
# #########################################################
def run_multiprocess_asynchronous_get(pvs):
    """Run asynchronous get with multiprocessing test."""
    def run_subprocess(pvs, pipe):
        pvsobj = []
        for pv in pvs:
            pvsobj.append(PV(pv))

        for pv in pvsobj:
            pv.wait_for_connection()

        while pipe.recv():
            out = []
            for pv in pvsobj:
                out.append(pv.timestamp)
            pipe.send(out)

    # create processes
    nrproc = 4

    # subdivide the pv list for the processes
    div = len(pvs) // nrproc
    rem = len(pvs) % nrproc
    sub = [div*i + min(i, rem) for i in range(nrproc+1)]

    procs = []
    my_pipes = []
    for i in range(nrproc):
        mine, theirs = Pipe()
        my_pipes.append(mine)
        pvsn = pvs[sub[i]:sub[i+1]]
        procs.append(CAProcess(
            target=run_subprocess,
            args=(pvsn, theirs),
            daemon=True))

    for proc in procs:
        proc.start()

    time.sleep(3)
    outt = []
    for i in range(600):
        t0 = time.time()
        for pipe in my_pipes:
            pipe.send(True)
        out = []
        for pipe in my_pipes:
            out.extend(pipe.recv())
        out = np.array(out)
        print(
            f'avg {(t0 - out.mean())*1000:.0f} ms    '
            f'std {out.std()*1000:.0f} ms    '
            f'p2p {(out.max()-out.min())*1000:.0f} ms    ')
        outt.append(out)
        time.sleep(1/50)

    np.savetxt('si_bpms.txt', outt)

    for pipe in my_pipes:
        pipe.send(False)
    for proc in procs:
        proc.join()


# #########################################################
# ####################### Analysis ########################
# #########################################################
def run_multiprocess_asynchronous_get_analysis():
    """Run analysis for asynchronous get with multiprocessing test."""
    si = np.loadtxt('si_bpms.txt') * 1000

    si2plt = si - si.mean(axis=1)[:, None]
    si2plt_std = si2plt.std(axis=1)
    si2plt_p2p = si2plt.max(axis=1) - si2plt.min(axis=1)
    si2plt2 = np.diff(si, axis=1)
    si2plt_bpm = si2plt2.std(axis=0)
    si2plt_bpm_max = si2plt2.max(axis=0)
    si2plt_bpm_min = si2plt2.min(axis=0)

    fig = plt.figure(figsize=(10, 10))
    gs = mgrid.GridSpec(4, 1, figure=fig)
    gs.update(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.25)
    ax = plt.subplot(gs[0, 0])
    ay = plt.subplot(gs[1, 0])
    az = plt.subplot(gs[2, 0])
    aw = plt.subplot(gs[3, 0])

    colors = mcmap.jet(np.linspace(0, 1, si.shape[1]))
    lines = ax.plot(si2plt, 'o-')
    ax.set_xlabel('aquisition number')
    ax.set_ylabel('timestamp - <timestamp> [ms]')
    for c, line in zip(colors, lines):
        line.set_color(c)

    ay.plot(si2plt_std, 'o-', label='STD')
    ay.plot(si2plt_p2p, 'o-', label='P2P')
    ay.set_xlabel('acquisition number')
    ay.set_ylabel('timestamp [ms]')
    ay.legend(loc='best')

    az.plot(si2plt_bpm, 'o-', label='STD')
    az.plot(si2plt_bpm_max, 'o-', label='MAX')
    az.plot(si2plt_bpm_min, 'o-', label='MIN')
    az.plot(si2plt_bpm_max - si2plt_bpm_min, 'o-', label='P2P')
    az.set_xlabel('BPM Index')
    az.set_ylabel('std(timestamp) [ms]')
    az.legend(loc='best')

    aw.hist(si2plt.ravel(), bins=100)
    aw.set_xlabel('timestamp [ms]')
    aw.set_ylabel('number of occurencies')

    plt.show()


# #########################################################
# #### multiprocess asynchronous monitor (monitoring) #####
# #########################################################
def run_multiprocess_asynchronous_monitor(pvs, total_time=1):
    """Run asynchronous get with multiprocessing test."""
    def run_subprocess(pvsi, pipe):
        def update_pv(pvname, value, **kwargs):
            nonlocal pvsinfo
            pvsinfo[pvname].append(kwargs['timestamp'])
            # pvsinfo[pvname].append(time.time())
            # pvsinfo[pvname].append(time.time()-kwargs['timestamp'])
        pvsobj = []
        pvsinfo = dict()
        for pvn in pvsi:
            pvsobj.append(PV(pvn))
            pvsinfo[pvn] = []

        for pvo in pvsobj:
            pvo.wait_for_connection()

        pipe.send(True)
        for pvo in pvsobj:
            pvo.add_callback(update_pv)
        pipe.recv()
        for pvo in pvsobj:
            pvo.clear_callbacks()
        pipe.send(pvsinfo)

    # create processes
    nrproc = min(4, len(pvs))

    # subdivide the pv list for the processes
    div = len(pvs) // nrproc
    rem = len(pvs) % nrproc
    sub = [div*i + min(i, rem) for i in range(nrproc+1)]

    procs = []
    my_pipes = []
    for i in range(nrproc):
        mine, theirs = Pipe()
        my_pipes.append(mine)
        pvsn = pvs[sub[i]:sub[i+1]]
        procs.append(CAProcess(
            target=run_subprocess,
            args=(pvsn, theirs),
            daemon=True))

    for proc in procs:
        proc.start()

    for pipe in my_pipes:
        pipe.recv()

    time.sleep(total_time)

    for pipe in my_pipes:
        pipe.send(False)
    pvsinfot = dict()
    for pipe in my_pipes:
        pvsinfot.update(pipe.recv())

    for proc in procs:
        proc.join()

    save_pickle('si_monitor_bpms', pvsinfot)


# #########################################################
# ####################### Analysis ########################
# #########################################################
def run_multiprocess_asynchronous_monitor_analysis():
    """Run analysis for asynchronous get with multiprocessing test."""
    pvsinfo = load_pickle('si_monitor_bpms')

    if len(pvsinfo) == 0:
        raise ValueError('No data loaded.')
    elif len(pvsinfo) == 1:
        run_camonitor_bpm_analysis(pvsinfo.popitem()[1])
        return

    pvs = sorted([SiriusPVName(pv) for pv in pvsinfo])

    fig = plt.figure(figsize=(8, 8))
    gs = mgrid.GridSpec(3, 1, figure=fig)
    gs.update(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.3)
    ax = plt.subplot(gs[0, 0])
    ay = plt.subplot(gs[1, 0], sharex=ax)
    az = plt.subplot(gs[2, 0])
    az.xaxis.set_major_formatter(MyFormatter(pvs))
    az.tick_params(axis="x", labelsize=8, rotation=0)
    ay.yaxis.set_major_formatter(MyFormatter(pvs))
    ay.tick_params(axis="y", labelsize=8, rotation=45)

    tims = []
    for tim in pvsinfo.values():
        tims.extend(tim)
    tims = np.sort(tims) * 1000
    reftime = tims[0]
    tims -= reftime
    # cumtims = np.arange(tims.size)

    # ax.plot(tims, cumtims)
    # ax.set_xlabel('$t_\mathrm{stamp}$ [ms]')
    # ax.set_ylabel(' cumulative # of evts')
    # ax.set_xlim([360, 500])
    bins = int(tims[-1]/5)
    ax.hist(tims, bins=bins)
    ax.set_xlabel('$t_\mathrm{stamp}$ [ms]')
    ax.set_ylabel('number of evts')

    freq = 25.16
    colors = mcmap.jet(np.linspace(0, 1, len(pvs)))
    for i, (cor, pvn) in enumerate(zip(colors, pvs)):
        val = np.array(pvsinfo[pvn])*1000 - reftime
        ay.errorbar(
            val, i + 0*val, yerr=0.5, marker='o', linestyle='', color=cor)
    for i in range(int(tims[-1]*freq/1000)+2):
        ay.axvline(
            x=(i - 0.5)/freq*1000, linewidth=1,
            color='k', linestyle='--')
    ay.set_xlabel('$t_\mathrm{stamp}$ [ms]')
    ay.set_ylabel('BPM Name')
    ay.grid(False)

    avg, std, mini, maxi = [], [], [], []
    for i, (cor, pvn) in enumerate(zip(colors, pvs)):
        val = np.array(pvsinfo[pvn])*1000 - reftime
        dtime = np.diff(val)
        avg.append(dtime.mean())
        std.append(dtime.std())
        mini.append(dtime.min())
        maxi.append(dtime.max())
    az.plot(avg, 'o-', label='AVG')
    az.plot(std, 'o-', label='STD')
    az.plot(maxi, 'o-', label='MAX')
    az.plot(mini, 'o-', label='MIN')
    az.plot(np.array(maxi) - np.array(mini), 'o-', label='P2P')
    az.set_xlabel('BPM Name')
    az.set_ylabel('Stats $\Delta t_\mathrm{stamp}$ [ms]')
    az.legend(loc='best', fontsize='xx-small')

    plt.show()


# #########################################################
# ######## multiprocess synchronous get optimized #########
# #########################################################
def run_multiprocess_synchronous_get_optimized(pvs):
    """Run synchronous get optimized with multiprocessing test."""
    def run_subprocess(pvs, pipe):
        pvschid = []
        for pv in pvs:
            chid = ca.create_channel(pv, connect=False, auto_cb=False)
            pvschid.append(chid)

        for chid in pvschid:
            ca.connect_channel(chid)
        time.sleep(5)
        for chid in pvschid:
            ftype = ca.promote_type(chid, use_time=True, use_ctrl=False)

        ca.poll()
        while pipe.recv():
            for chid in pvschid:
                ca.get_with_metadata(chid, ftype=ftype, wait=False)
            out = []
            for chid in pvschid:
                data = ca.get_complete_with_metadata(chid, ftype=ftype)
                out.append(data['timestamp'])
            pipe.send(out)

    proc = psutil.Process()

    # create processes
    nrproc = 2

    # subdivide the pv list for the processes
    div = len(pvs) // nrproc
    rem = len(pvs) % nrproc
    sub = [div*i + min(i, rem) for i in range(nrproc+1)]

    procs = []
    my_pipes = []
    for i in range(nrproc):
        mine, theirs = Pipe()
        my_pipes.append(mine)
        pvsn = pvs[sub[i]:sub[i+1]]
        procs.append(CAProcess(
            target=run_subprocess,
            args=(pvsn, theirs),
            daemon=True))

    for proc in procs:
        proc.start()

    outt = []
    for i in range(600):
        t0 = time.time()
        for pipe in my_pipes:
            pipe.send(True)
        out = []
        for pipe in my_pipes:
            out.extend(pipe.recv())
        out = np.array(out)
        t1 = time.time()
        print(
            f'avg {(t1 - out.mean())*1000:.0f} ms    '
            f'std {out.std()*1000:.0f} ms    '
            f'p2p {(out.max()-out.min())*1000:.0f} ms    ')
        outt.append(out)
        dtime = t1 - t0
        dtime -= 1/30
        if dtime < 0:
            time.sleep(-dtime)
        else:
            print(f'loop exceeded {dtime*1000:.1}ms.')

    np.savetxt('si_sync_bpms.txt', outt)

    for pipe in my_pipes:
        pipe.send(False)
    for proc in procs:
        proc.join()


# #########################################################
# #################### Test EpicsOrbit ####################
# #########################################################
def run_test_epicsorbit_class(total_time=30):
    """Test EpicsOrbit class."""
    def callback(pvname, value, **kwargs):
        if not pvname.endswith('SlowOrbX-Mon'):
            return
        nonlocal t0, orbits
        t1 = time.time()
        value = t1 - np.asarray(value) * 1e3
        orbits.append(value)
        t0 = t1

    orb = EpicsOrbit('SI')

    time.sleep(5)
    print('Setting orbit acquisition rate')
    t0 = time.time()
    orbits = []
    orb.add_callback(callback)

    orb.set_orbit_mode(orb._csorb.SOFBMode.SlowOrb)
    orb.set_orbit_acq_rate(40)

    for i in range(total_time):
        print(f'remaining time {total_time - i:5.1f} s', end='\r')
        time.sleep(1)
    print('\nDone!')

    save_pickle('epics_orbit', np.asarray(orbits))


# #########################################################
# ####################### Analysis ########################
# #########################################################
def run_test_epicsorbit_analysis():
    """Run analysis for epicsorbit test."""
    orbits = load_pickle('epics_orbit')*1000
    orbits_avg = orbits.mean(axis=1)
    orbits_std = orbits.std(axis=1)
    orbits_p2p = orbits.max(axis=1) - orbits.min(axis=1)

    fig = plt.figure(figsize=(10, 10))
    gs = mgrid.GridSpec(3, 1, figure=fig)
    gs.update(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.25)
    ax = plt.subplot(gs[0, 0])
    ay = plt.subplot(gs[1, 0])
    az = plt.subplot(gs[2, 0])

    colors = mcmap.jet(np.linspace(0, 1, orbits.shape[1]))
    lines = ax.plot(orbits, 'o-')
    ax.set_xlabel('aquisition number')
    ax.set_ylabel(r'$\Delta t_\mathrm{stamp}$ [ms]')
    for c, line in zip(colors, lines):
        line.set_color(c)
    ax.set_ylim([-5, 30])

    ay.plot(orbits_avg, 'o-', label='AVG')
    ay.plot(orbits_std, 'o-', label='STD')
    ay.plot(orbits_p2p, 'o-', label='P2P')
    ay.set_xlabel('acquisition number')
    ay.set_ylabel(r'$t_\mathrm{stamp}$ [ms]')
    ay.legend(loc='best')
    ay.set_ylim([-5, 30])

    az.hist(orbits.ravel(), bins=100, range=(0, 30))
    az.set_xlabel(r'$t_\mathrm{stamp}$ [ms]')
    az.set_ylabel('number of evts')

    plt.show()


# #########################################################
# #################### Test SOFB Orbit ####################
# #########################################################
def run_test_sofb():
    """Test SOFB IOC frequency."""
    def print_time(pvname, value, **kwargs):
        _ = pvname
        nonlocal times, values
        # times.append(time.time())
        times.append(kwargs['timestamp'])
        values.append(value[0])
        print(datetime.datetime.fromtimestamp(times[-1]).isoformat(), value[0])

    times = []
    values = []

    pv = PV('SI-Glob:AP-SOFB:SlowOrbX-Mon')
    pv.wait_for_connection()
    pv.add_callback(print_time)

    total = 30
    time.sleep(total)

    # times = values
    print(f'frequency:    {len(times)/total:.2f} Hz')
    print(f'average time: {np.mean(np.diff(times))*1000:.2f} ms')
    print(f'std time:     {np.std(np.diff(times))*1000:.2f} ms')
    print(f'min time:     {np.min(np.diff(times))*1000:.2f} ms')
    print(f'max time:     {np.max(np.diff(times))*1000:.2f} ms')
    np.savetxt('si_sofb.txt', times)
    pv.clear_callbacks()


# #########################################################
# ############## Analysis of Test SOFB Orbit ##############
# #########################################################
def run_test_sofb_analysis():
    """Run analysis of SOFB IOC Test."""
    times = np.loadtxt('si_sofb.txt') * 1000
    dtimes = np.diff(times)
    # dtimes = times
    dt_avg = dtimes.mean()
    dt_std = dtimes.std()
    dt_min = dtimes.min()
    dt_max = dtimes.max()
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

    vals = [dt_std, dt_min, dt_avg, dt_p2p, dt_max]
    x = np.arange(len(vals))
    ay.bar(x, vals)
    ay.set_xticklabels(('', 'STD', 'MIN', 'AVG', 'P2P', 'MAX'))
    # ay.set_xlabel('Stats')
    ay.set_ylabel('dtime [ms]')

    az.hist(dtimes, bins=100)
    az.set_xlabel('dtime [ms]')
    az.set_ylabel('number of occurencies')

    plt.show()


# #########################################################
# ############### Analysis CAMonitor of BPM ###############
# #########################################################
def run_camonitor_bpm_analysis(tstamp=None):
    """Run analysis of camonitor of BPM."""
    def fit_gauss(binc, den, A, x0, sig):
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma*sigma))

        coeff, var_matrix = curve_fit(
            gauss, binc, den, p0=[A, x0, sig])

        # Get the fitted curve
        return gauss(binc, *coeff), coeff

    if tstamp is None:
        with open('bpm01m2.txt', 'r') as fil:
            data = fil.readlines()

        tstamp = []
        for line in data:
            tim = line.split()[2]
            tim = datetime.datetime.strptime(tim, '%H:%M:%S.%f')
            tstamp.append(tim.timestamp())
        tstamp = np.array(tstamp)

    dtimes = np.diff(tstamp) * 1000
    dt_avg = dtimes.mean()
    dt_std = dtimes.std()
    dt_min = dtimes.min()
    dt_max = dtimes.max()
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

    vals = [dt_std, dt_min, dt_avg, dt_p2p, dt_max]
    x = np.arange(len(vals))
    ay.bar(x, vals)
    ay.set_xticklabels(('', 'STD', 'MIN', 'AVG', 'P2P', 'MAX'))
    # ay.set_xlabel('Stats')
    ay.set_ylabel('dtime [ms]')

    dens, bins, _ = az.hist(dtimes, bins=100)
    az.set_xlabel('dtime [ms]')
    az.set_ylabel('number of occurencies')

    bins += (bins[1]-bins[0])/2
    bins = bins[:-1]
    lower = (bins > 25) & (bins < 44)
    dens1 = dens[lower]
    bins1 = bins[lower]
    fitb1, coeff1 = fit_gauss(
        bins1, dens1, dens1.max(), bins1.mean(), dens1.std())
    az.plot(bins1, fitb1, 'tab:orange')
    txt = r'$\mu$ = '+f'{coeff1[1]:.1f} ms\n'
    txt += r'$\sigma$ = '+f'{coeff1[2]:.1f} ms'
    az.annotate(
        txt, xy=(coeff1[1], coeff1[0]),
        xytext=(coeff1[1] - 3*coeff1[2], coeff1[0]*0.6),
        horizontalalignment='right',
        arrowprops=dict(facecolor='black', shrink=0.05))

    upper = bins > 44
    dens2 = dens[upper]
    bins2 = bins[upper]
    if dens2.max() > 4:
        fitb2, coeff2 = fit_gauss(
            bins2, dens2, dens2.max(), bins2.mean(), dens2.std())
        az.plot(bins2, fitb2, 'tab:orange')
        txt = r'$\mu$ = '+f'{coeff2[1]:.1f} ms\n'
        txt += r'$\sigma$ = '+f'{coeff2[2]:.1f} ms'
        az.annotate(
            txt, xy=(coeff2[1], coeff2[0]),
            xytext=(coeff2[1] + 3*coeff2[2], coeff2[0]),
            arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()


if __name__ == '__main__':
    # ##### Create the list of PVs to connect #####
    sofb = SOFBFactory.create('SI')
    bpms = []
    bpms.extend(sofb.bpm_names)
    # bpms[:1]

    # bpms = []
    # sofb = SOFBFactory.create('TB')
    # bpms.extend(sofb.bpm_names)
    # sofb = SOFBFactory.create('TS')
    # bpms.extend(sofb.bpm_names)
    # # bpms = bpms[:1]

    pvsraw = []
    for bpm in bpms:
        pvsraw.append(bpm+':PosX-Mon')

    secs = ()
    if secs:
        pvs = []
        for pvn in pvsraw:
            pvn = SiriusPVName(pvn)
            if pvn.sub.startswith(secs):
                pvs.append(pvn)
    else:
        pvs = pvsraw

    # ##### Select a test to run #####

    # run_synchronous_get_optimized(pvs)

    # run_multiprocess_synchronous_get_optimized(pvs)

    # run_multiprocess_asynchronous_get(pvs)
    # run_multiprocess_asynchronous_get_analysis()

    # run_multiprocess_asynchronous_monitor(pvs, total_time=0.4)
    # run_multiprocess_asynchronous_monitor_analysis()

    # run_test_epicsorbit_class(total_time=5)
    # run_test_epicsorbit_analysis()

    # run_test_sofb()
    # run_test_sofb_analysis()
