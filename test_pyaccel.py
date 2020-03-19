#!/usr/bin/env python-sirius

import sys
import numpy as np
import pyaccel

ele = pyaccel.elements.matrix('test', 0)
mat = ele.matrix66
#mat[0] = (1, 3, 0, 0, 0, 0)
#mat[1] = (-2, 1-2*3, 0, 0, 0, 0)
#ele.matrix66 = mat
#print(ele.matrix66)
ele.KxL = 2
ele.KyL = -2
#ele.KsyL = 3
#ele.KsxL = 3
print(ele.matrix66)
ele.length = 0.1 
ele.nr_steps = 10
dr = pyaccel.elements.drift('dr', 0.0)
quad = pyaccel.elements.quadrupole('quad', 0.1, 0)
#quad.KsL = 3
quad.KL = 2
quad.nr_steps = 10

acc = pyaccel.accelerator.Accelerator(energy=3e9)
acc.append(ele)
acc.append(dr)
acc.radiation_on = False
acc2 = pyaccel.accelerator.Accelerator(energy=3e9)
acc2.append(quad)
acc2.append(dr)
acc2.radiation_on = False

inn = np.array([1.0, 1, 1, 1, 0, 0])

out, *_ = pyaccel.tracking.linepass(acc, inn, indices='closed')
out2, *_ = pyaccel.tracking.linepass(acc2, inn, indices='closed')

for o, o2 in zip(out.T, out2.T):
    print('turn'.center(40, '-'))
    print(o[0], o[1], o[2], o[3])
    print(o2[0], o2[1], o2[2], o2[3])

