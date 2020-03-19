#!/usr/bin/env python-sirius

import siriuspy.clientconfigdb as config

conn = config.ConfigDBClient(config_type='global_config')

conf_names = ['temp', 'test', 'mb_1scrnBO_TB_neworb_corr_newPMdelays', 'mb_1scrnBO_TB_orb_corr_NewPMDelays']                 

for conf in conf_names: 
    val = conn.get_config_value(conf) 
    val2 = dict(pvs=[]) 
    for name, v, d in val['pvs']:  
        if name.startswith('RA-RaMO:TI-EVG'):  
            name = 'AS' + name[2:]  
        val2['pvs'].append([name, v, d]) 
    conn.insert_config(conf, val2) 
