#!/usr/bin/env python-sirius
"""IOC Module."""

import os as _os
import logging as _log
import signal as _signal
import time as _time
from copy import deepcopy as _dcopy
from threading import Thread

import numpy as np
import pcaspy as _pcaspy
import pcaspy.tools as _pcaspy_tools

import siriuspy.util as _util


def _stop_now(signum, frame):
    _log.info('SIGNAL received')
    global STOP_EVENT
    STOP_EVENT = True


def _attribute_access_security_group(server, db):
    for k, v in db.items():
        if k.endswith(('-RB', '-Sts', '-Cte', '-Mon')):
            v.update({'asg': 'rbpv'})
    path_ = _os.path.abspath(_os.path.dirname(__file__))
    server.initAccessSecurityFile(path_ + '/access_rules.as')


STOP_EVENT = False

PVTMPL = lambda pref, pvi, suf: f'IOC{pref:02d}:PV{pvi:03d}-{suf:s}'
_DBASE = {'type': 'float', 'value': 0.0, 'count': 8}
# _DBASE['asyn'] = True
PVSDB = dict()
for pref in range(40):
    for pvi in range(40):
        PVSDB[PVTMPL(pref, pvi, 'SP')] = _dcopy(_DBASE)
        PVSDB[PVTMPL(pref, pvi, 'RB')] = _dcopy(_DBASE)


def get_pvsdb(index=None):
    """."""
    if index is None:
        return PVSDB
    prf = f'IOC{index:02d}'
    return {pv: db for pv, db in PVSDB.items() if pv.startswith(prf)}


class _PCASDriver(_pcaspy.Driver):

    def __init__(self, index=None):
        super().__init__()
        self.index = index
        self.pvs2monitor = list(range(20, 40))
        self.thr = Thread(target=self._monitor)
        self.thr.daemon = True

    def write(self, reason, value):
        """."""
        # self._update(reason, value)
        th = Thread(target=self._update, args=(reason, value))
        th.daemon = True
        th.start()
        return True

    def _monitor(self):
        index = self.index or 0
        for pvi in range(self.pvs2monitor):
            reason = PVTMPL(index, pvi, 'RB')
            val = np.random.rand(8)
            self.update_pv(reason, val)

    def start_thread(self):
        """."""
        self.thr.start()

    def _update(self, reason, value):
        if not self._isValid(reason, value):
            return False
        self.update_pv(reason, value)
        _time.sleep(0.02)
        reason2 = reason.replace('-SP', '-RB')
        self.update_pv(reason2, value)
        # self.callbackPV(reason)

    def update_pv(self, pvname, value, **kwargs):
        """."""
        self.setParam(pvname, value)
        self.updatePV(pvname)

    def _isValid(self, reason, val):
        if reason.endswith(('-Sts', '-RB', '-Mon', '-Cte')):
            _log.debug('PV {0:s} is read only.'.format(reason))
            return False
        if val is None:
            msg = 'client tried to set None value. refusing...'
            _log.error(msg)
            return False
        enums = self.getParamInfo(reason, info_keys=('enums', ))['enums']
        if enums and isinstance(val, int) and val >= len(enums):
            _log.warning('value %d too large for enum type PV %s', val, reason)
            return False
        return True


def run(index=None, debug=False):
    """Start the IOC."""
    _util.configure_log_file(debug=debug)
    _log.info('Starting...')

    # define abort function
    _signal.signal(_signal.SIGINT, _stop_now)
    _signal.signal(_signal.SIGTERM, _stop_now)

    # Creates App object
    _log.debug('Creating SOFB Object.')
    # create a new simple pcaspy server and driver to respond client's requests
    _log.info('Creating Server.')
    server = _pcaspy.SimpleServer()
    pvsdb = get_pvsdb(index)
    _attribute_access_security_group(server, pvsdb)
    _log.info('Setting Server Database.')
    server.createPV('', pvsdb)
    _log.info('Creating Driver.')
    driver = _PCASDriver(index=index)

    # initiate a new thread responsible for listening for client connections
    server_thread = _pcaspy_tools.ServerThread(server)
    server_thread.setDaemon(True)
    _log.info('Starting Server Thread.')
    server_thread.start()

    # driver.start_thread()

    # main loop
    while not STOP_EVENT:
        _time.sleep(5)

    _log.info('Stoping Server Thread...')
    # sends stop signal to server thread
    server_thread.stop()
    server_thread.join()
    _log.info('Server Thread stopped.')
    _log.info('Good Bye.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Set RF frequency.")
    parser.add_argument(
        'index', type=int, help='.', choices=list(range(40)))
    parser.add_argument(
        '-a', '--all', action='store_true', default=False,
        help='. [Hz]')
    args = parser.parse_args()

    if args.all:
        run()
    else:
        run(args.index)
