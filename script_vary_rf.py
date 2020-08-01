#!/usr/bin/env python3

import sys
import time

import epics
import numpy as np

RF_DELTA_MIN = 0.1  # [Hz]
RF_DELTA_MAX = 15000.0  # [Hz]
RF_DELTA_RMP = 20  # [Hz]


def set_rf_frequency(pvobj, value, delta=False):
    if not pvobj.connected:
        raise ValueError('PV not connected.')
    freq0 = pvobj.value
    if freq0 is None:
        raise ValueError('Could not get current value from PV')
    if delta:
        print('Interpreting new value as a delta frequency:')
        value += freq0
    delta = abs(value-freq0)
    if delta < RF_DELTA_MIN or delta > RF_DELTA_MAX:
        raise ValueError(
            'Delta frequency out of range: '
            f'({RF_DELTA_MIN:.1f}, {RF_DELTA_MAX:.1f})')
    npoints = int(round(delta/RF_DELTA_RMP)) + 2
    freq_span = np.linspace(freq0, value, npoints)[1:]
    print(f'Initial value: {freq0:.1f}')
    if freq_span.size:
        print(f'Applying delta in {freq_span.size:d} steps')
    for freq in freq_span:
        print('.', end='')
        pvobj.put(freq, wait=False)
        time.sleep(1.0)
    if freq_span.size:
        print()
    print('Done!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Set RF frequency.")
    parser.add_argument('freq', type=float, help='The new value to be set.')
    parser.add_argument(
        '-d', '--delta', action='store_true', default=False,
        help=(
            'If present, value given for frequency will be intepreted as a '
            'frequency variation referenced around the current value of '
            'frequency. [Hz]'))
    args = parser.parse_args()

    pv_rfgen = epics.PV('RF-Gen:GeneralFreq-SP')
    pv_rfgen.wait_for_connection()

    set_rf_frequency(pv_rfgen, args.freq, delta=args.delta)
