#!/usr/bin/env python-sirius

import time
from epics import PV
from siriuspy.search import PSSearch, LLTimeSearch

cycle = PV('AS-RaMO:TI-EVG:CycleExtTrig-Cmd')


def put_pv(sp, rb, value):
    sp.value = value
    while rb.value != value:
        time.sleep(0.1)


def test_trigger(trig_sp, trig_rb, pvs_ps, trigs, trig):
    print('{}'.format(trig_sp.pvname.replace('State-Sel', '')))
    vals0 = [pv.get(use_monitor=False) for pv in pvs_ps]
    put_pv(trig_sp, trig_rb, 1)
    cycle.value = 1
    time.sleep(0.5)
    put_pv(trig_sp, trig_rb, 0)
    time.sleep(1)
    vals1 = [pv.get(use_monitor=False) for pv in pvs_ps]

    pss = set()
    for v0, v1, pv in zip(vals0, vals1, pvs_ps):
        if v0 != v1:
            pss.add(pv.pvname.replace(':WfmSyncPulseCount-Mon', ''))

    for ps in pss:
        print('    {}'.format(ps))
    amais = pss - trigs[trig]
    for name in amais:
        print('    a mais: {}'.format(name))
    amenos = trigs[trig] - pss
    for name in amenos:
        print('    a menos: {}'.format(name))


if __name__ == '__main__':

    pss = PSSearch.get_psnames(
        {'sec': 'BO', 'dis': 'PS', 'sub': '[0-9]{2}.*',})
    pss.extend(PSSearch.get_psnames(
        {'sec': 'SI', 'dis': 'PS', 'sub': '[0-9]{2}.*', 'dev': '[QC]'}))

    trigs = dict()
    for i, ps in enumerate(pss):
        chan = ps.substitute(propty_name='BCKPLN')
        trig = LLTimeSearch.get_trigger_name(chan)
        if trig in trigs:
            trigs[trig].add(ps)
        else:
            trigs[trig] = {ps, }

    pvs_ps = list()
    for ps in sorted(pss):
        pvs_ps.append(PV(ps.substitute(
            propty_name='WfmSyncPulseCount', propty_suffix='Mon')))

    pvs_trig_sp = dict()
    pvs_trig_rb = dict()
    for trig in trigs:
        pvn = LLTimeSearch.get_channel_output_port_pvname(trig)
        pvs_trig_sp[trig] = PV(pvn.substitute(
            propty_name=pvn.propty+'State', propty_suffix='Sel'))
        pvs_trig_rb[trig] = PV(pvn.substitute(
            propty_name=pvn.propty+'State', propty_suffix='Sts'))

    ps_conn = all(map(lambda x: x.wait_for_connection(2), pvs_ps))
    tr_conn_sp = all(map(
        lambda x: x.wait_for_connection(2), pvs_trig_sp.values()))
    tr_conn_rb = all(map(
        lambda x: x.wait_for_connection(2), pvs_trig_rb.values()))
    print(ps_conn, tr_conn_sp, tr_conn_rb, cycle.connected)

    for trig in sorted(trigs):
        if not trig.startswith('IA-10RaBPM:TI-AMCFPGAEVR'):
            continue
        put_pv(pvs_trig_sp[trig], pvs_trig_sp[trig], 0)

    for trig in sorted(trigs):
        if not trig.startswith('IA-10RaBPM:TI-AMCFPGAEVR'):
            continue
        test_trigger(pvs_trig_sp[trig], pvs_trig_sp[trig], pvs_ps, trigs, trig)
