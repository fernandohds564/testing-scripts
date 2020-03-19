import numpy as np

def _calc_poly(p, o):
    o2 = o*o
    o4 = o2*o2
    o6 = o4*o2
    o8 = o4*o4
    p2 = p*p
    p3 = p2*p
    p5 = p3*p2
    p7 = p5*p2
    p9 = p7*p2

    return (  
        p   *(8.57433e+06 + o2*4.72785e+06 + o4*4.03599e+06 + o6*2.81406e+06 + o8*9.67341e+06) +
        p3*(4.01544e+06 + o2*1.05649e+07 + o4*9.85821e+06 + o6*8.6841e+07) +
        p5*(3.94658e+06 + o2*5.27686e+06 + o4*2.28462e+08) +
        p7*(-1.1398e+06 + o2*9.5492e+07) +
        p9*2.43619e+07)


def calc_poly(x, y):

    x /= 8.57433e+06
    y /= 8.57433e+06

    xf = _calc_poly(x,y)
    yf = _calc_poly(y,x)

    return xf, yf
