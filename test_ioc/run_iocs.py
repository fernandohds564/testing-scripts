#!/usr/bin/env python-sirius

import subprocess as _subp
import signal as _signal

STOP_EVENT = False


def _stop_now(signum, frame):
    """."""
    for prc in procs:
        prc.send_signal(signum)


# define abort function
_signal.signal(_signal.SIGINT, _stop_now)
_signal.signal(_signal.SIGTERM, _stop_now)

procs = []
for i in range(40):
    print(f'launching process {i:02d}')
    procs.append(_subp.Popen(
        f'python-sirius test_ioc.py {i:d}'.split(),
        stdout=_subp.PIPE,
        stderr=_subp.PIPE))

for i, proc in enumerate(procs):
    print(f'waiting process {i:02d}')
    proc.wait()
