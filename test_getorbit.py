#!/usr/bin/env python-sirius

# ----------- synchronous get optimized ------

# import time
# import psutil

# from siriuspy.sofb.csdev import SOFBFactory
# from epics import ca, PV

# proc = psutil.Process()
# sofb = SOFBFactory.create('SI')

# pvs = []
# for bpm in sofb.bpm_names:
#     pvs.append(bpm+':PosX-Mon')
# for bpm in sofb.bpm_names:
#     pvs.append(bpm+':PosY-Mon')

# pvschid = []
# for pv in pvs:
#     chid = ca.create_channel(pv, connect=False, auto_cb=False)
#     pvschid.append(chid)

# for chid in pvschid:
#     ca.connect_channel(chid)

# ca.poll()
# for i in range(600):
#     t0 = time.time()
#     for chid in pvschid:
#         ca.get(chid, wait=False)
#     out = []
#     for chid in pvschid:
#         out.append(ca.get_complete(chid))
#     print(f'dtime {(time.time()-t0)*1000:.1f}ms   pos {out[0]:.0f}nm')

# ----------- asynchronous get (monitoring)------

# import time

# from siriuspy.sofb.csdev import SOFBFactory
# from epics import ca, PV

# proc = psutil.Process()
# sofb = SOFBFactory.create('SI')

# pvs = []
# for bpm in sofb.bpm_names:
#     pvs.append(bpm+':PosX-Mon')
#     pvs.append(bpm+':PosY-Mon')

# for pv in pvs:
#     pvs.append(PV(pv))

# for pv in pvs:
#     pvs.wait_for_connection()

# for i in range(600):
#     t0 = time.time()
#     out = []
#     for pv in pvs:
#         out.append(pv.value)
#     time.sleep(0.03)
#     print(f'dtime {(time.time()-t0)*1000:.1f}ms   pos {out[0]:.0f}nm')


# ----------- multiprocess asynchronous get (monitoring)------

import time
from multiprocessing import Pipe
import numpy as np

from epics import PV, CAProcess
from siriuspy.sofb.csdev import SOFBFactory


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


sofb = SOFBFactory.create('SI')
bpm_names = list(sofb.bpm_names)

pvs = []
for bpm in bpm_names:
    pvs.append(bpm+':PosX-Mon')
    pvs.append(bpm+':PosY-Mon')


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
    time.sleep(1/30)

np.savetxt('si_bpms.txt', outt)

for pipe in my_pipes:
    pipe.send(False)
for proc in procs:
    proc.join()

# ---------- Analysis ---------------

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as mcmap
# import matplotlib.gridspec as mgrid
# from matplotlib import rcParams
# rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'axes.grid': True})

# si = np.loadtxt('si_bpms.txt') * 1000

# si2plt = si - si.mean(axis=1)[:, None]
# si2plt_std = si2plt.std(axis=1)
# si2plt_p2p = si2plt.max(axis=1) - si2plt.min(axis=1)
# si2plt_bpm = si2plt.std(axis=0)
# si2plt_bpm_max = si2plt.max(axis=0)
# si2plt_bpm_min = si2plt.min(axis=0)

# fig = plt.figure(figsize=(15, 15))
# gs = mgrid.GridSpec(4, 1, figure=fig)
# gs.update(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.25)
# ax = plt.subplot(gs[0, 0])
# ay = plt.subplot(gs[1, 0])
# az = plt.subplot(gs[2, 0])
# aw = plt.subplot(gs[3, 0])

# colors = mcmap.jet(np.linspace(0, 1, si.shape[1]))
# lines = ax.plot(si2plt, 'o-')
# ax.set_xlabel('aquisition number')
# ax.set_ylabel('timestamp - <timestamp> [ms]')
# for c, line in zip(colors, lines):
#     line.set_color(c)

# ay.plot(si2plt_std, 'o-', label='STD')
# ay.plot(si2plt_p2p, 'o-', label='P2P')
# ay.set_xlabel('acquisition number')
# ay.set_ylabel('timestamp [ms]')
# ay.legend(loc='best')

# az.plot(si2plt_bpm, 'o-', label='STD')
# az.plot(si2plt_bpm_max, 'o-', label='MAX')
# az.plot(si2plt_bpm_min, 'o-', label='MIN')
# az.plot(si2plt_bpm_max - si2plt_bpm_min, 'o-', label='P2P')
# az.set_xlabel('BPM')
# az.set_ylabel('std(timestamp) [ms]')
# az.legend(loc='best')

# aw.hist(si2plt.ravel(), bins=100)
# aw.set_xlabel('timestamp [ms]')
# aw.set_ylabel('number of occurencies')

# plt.show()

# ----------- multiprocess synchronous get optimized------

# import time
# from multiprocessing import Pipe
# import psutil
# import numpy as np

# from epics import CAProcess, ca, PV
# from siriuspy.sofb.csdev import SOFBFactory


# def run_subprocess(pvs, pipe):
#     pvschid = []
#     for pv in pvs:
#         chid = ca.create_channel(pv, connect=False, auto_cb=False)
#         pvschid.append(chid)

#     for chid in pvschid:
#         ca.connect_channel(chid)
#     time.sleep(5)
#     for chid in pvschid:
#         ftype = ca.promote_type(chid, use_time=True, use_ctrl=False)

#     ca.poll()
#     while pipe.recv():
#         for chid in pvschid:
#             ca.get_with_metadata(chid, ftype=ftype, wait=False)
#         out = []
#         for chid in pvschid:
#             data = ca.get_complete_with_metadata(chid, ftype=ftype)
#             out.append(data['timestamp'])
#         pipe.send(out)


# proc = psutil.Process()

# sofb = SOFBFactory.create('SI')
# bpm_names = list(sofb.bpm_names[7:15])

# pvs = []
# for bpm in bpm_names:
#     pvs.append(bpm+':PosX-Mon')
#     pvs.append(bpm+':PosY-Mon')

# # create processes
# nrproc = 2

# # subdivide the pv list for the processes
# div = len(pvs) // nrproc
# rem = len(pvs) % nrproc
# sub = [div*i + min(i, rem) for i in range(nrproc+1)]

# procs = []
# my_pipes = []
# for i in range(nrproc):
#     mine, theirs = Pipe()
#     my_pipes.append(mine)
#     pvsn = pvs[sub[i]:sub[i+1]]
#     procs.append(CAProcess(
#         target=run_subprocess,
#         args=(pvsn, theirs),
#         daemon=True))

# for proc in procs:
#     proc.start()

# outt = []
# for i in range(600):
#     t0 = time.time()
#     for pipe in my_pipes:
#         pipe.send(True)
#     out = []
#     for pipe in my_pipes:
#         out.extend(pipe.recv())
#     out = np.array(out)
#     t1 = time.time()
#     print(
#         f'avg {(t1 - out.mean())*1000:.0f} ms    '
#         f'std {out.std()*1000:.0f} ms    '
#         f'p2p {(out.max()-out.min())*1000:.0f} ms    ')
#     outt.append(out)
#     dtime = t1 - t0
#     dtime -= 1/30
#     if dtime < 0:
#         time.sleep(-dtime)
#     else:
#         print(f'loop exceeded {dtime*1000:.1}ms.')

# np.savetxt('si_sync_bpms.txt', outt)

# for pipe in my_pipes:
#     pipe.send(False)
# for proc in procs:
#     proc.join()

# ---------- Test with EpicsOrbit ------------

# import time
# import numpy as np
# from siriuspy.sofb.orbit import EpicsOrbit


# orb = EpicsOrbit('SI')

# time.sleep(5)
# print('Setting orbit acquisition rate')
# orb.set_orbit_acq_rate(30)
# orb.set_orbit_mode(orb._csorb.SOFBMode.SlowOrb)

# t0 = time.time()
# def callback(pvname, value, **kwargs):
#     if not pvname.endswith('OrbX-Mon'):
#         return
#     global t0
#     t1 = time.time()
#     # print(
#     #     f'dtime {(t1-t0)*1000:.1f} ms    '
#     #     f'len {len(value):d}   first {value[0]:.0f} nm    '
#     #     f'avg {np.mean(value):.0f} nm    std {np.std(value):.0f} nm')
#     print(
#         f'dtime {(t1-t0)*1000:.1f} ms    '
#         f'avg {(t1 - np.mean(value))*1000:.0f} ms    '
#         f'std {np.std(value)*1000:.0f} ms    '
#         f'p2p {(np.max(value)-np.min(value))*1000:.0f} ms    ')
#     t0 = t1

# orb.add_callback(callback)

# for i in range(60):
#    time.sleep(1)
