#!/usr/bin/env python-sirius

import sys as _sys
import logging as _log
from siriuspy.timesys.static_table import create_static_table

# logfile = 'test.txt'
logfile = '/home/sirius/repos/control-system-constants/timesys/timing-devices-connection.txt'

if __name__ == '__main__':
    #create_static_table(
    #    fname='Downloads/Cabos_e_Fibras_Sirius.xlsx', local=True, logfile=logfile)
    create_static_table(logfile=logfile)

