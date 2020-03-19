#!/usr/local/env python3

import urllib.request as _urllib_request
import ast

#SERVER_ADDR = 'http://10.128.254.203'
SERVER_ADDR = 'http://10.0.38.42/control-system-constants/'
TIMEOUT = 1.0

def high_level_events():
    """Return the data defining the high level events."""
    try:
        url = SERVER_ADDR + '/timesys/high-level-events.py'
        response = _urllib_request.urlopen(url, timeout=TIMEOUT)
        data = response.read()
        text = data.decode('utf-8')
    except Exception:
        errtxt = 'Error reading url "' + url + '"!'
        raise Exception(errtxt)
    return text

HLEvents = ast.literal_eval(high_level_events())
print(HLEvents)
