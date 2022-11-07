# %%
import time

import numpy as np
import scipy.linalg as scylin

import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs

path = '/home/imas/repos/id-sabia/model-03/simulation/magnetic/'
path += 'fieldmaps_and_kickmaps/'

fname = '2021-03-14_DeltaSabia_'
fname += 'EllipticalPolarization_'
fname += 'Kh=2.23_Kv=5.58_X=-7_7mm_Y=-4_4mm_Z=-1000_1000mm'
fname += '.fld'

X, Y, Z, Bx, By, Bz = np.loadtxt(path+fname, skiprows=21, unpack=True)

# %%

x = np.sort(list({x for x in X}))
y = np.sort(list({y for y in Y}))
z = np.sort(list({z for z in Z}))

X = X.reshape((z.size, y.size, x.size))
Y = Y.reshape((z.size, y.size, x.size))
Z = Z.reshape((z.size, y.size, x.size))
Bx = Bx.reshape((z.size, y.size, x.size))
By = By.reshape((z.size, y.size, x.size))
Bz = Bz.reshape((z.size, y.size, x.size))

# %%

idx = np.argmin(np.abs(x))
idy = np.argmin(np.abs(y))
idz = np.argmin(np.abs(z))

fig = mplt.figure(figsize=(6, 5))
gs = mgs.GridSpec(1, 1, left=0.12, right=0.98, bottom=0.12)

ax = fig.add_subplot(gs[0, 0])

ax.plot(z, Bx[:, idy, idx])
# ax.plot(y, Y[0, :, 0])
# ax.plot(x, X[0, 0, :])
mplt.show()

# %%


def field_by_on_axis(kx, kz, cmat, phimat, x, y, z):
    bx = np.zeros(x.shape, dtype=float)
    by = np.zeros(x.shape, dtype=float)
    bz = np.zeros(x.shape, dtype=float)
    for nx in range(cmat.shape[0]):
        for nz in range(1, cmat.shape[1]):
            kxn = nx*kx
            kzn = nz*kz
            kyn = np.sqrt(kxn*kxn + kzn*kzn)
            cosx = np.cos(kxn*x)
            cosy = np.cosh(kyn*y)
            cosz = np.cos(kzn*z + phimat[nx, nz])

            byn = cmat[nx, nz] * cosx
            byn *= cosy
            byn *= cosz

            sinx = np.sin(kxn*x)
            siny = np.sinh(kyn*y)

            bxn = cmat[nx, nz] * sinx
            bxn *= kxn / kyn
            bxn *= siny
            bxn *= cosz

            sinz = np.sin(kzn*z + phimat[nx, nz])

            bzn = cmat[nx, nz] * cosx
            bzn *= kzn / kyn
            bzn *= siny
            bzn *= sinz

            bx += bxn
            by += bzn
            bz += bzn
    return bx, by, bz


def field_bx_on_axis(ky, kz, cmat, phimat, x, y, z):
    bx = np.zeros(x.shape, dtype=float)
    by = np.zeros(x.shape, dtype=float)
    bz = np.zeros(x.shape, dtype=float)
    for ny in range(cmat.shape[0]):
        for nz in range(1, cmat.shape[1]):
            kyn = ny*ky
            kzn = nz*kz
            kxn = np.sqrt(kyn*kyn + kzn*kzn)
            cosx = np.cosh(kxn*x)
            cosy = np.cos(kyn*y)
            cosz = np.cos(kzn*z + phimat[nx, nz])

            bxn = cmat[nx, nz] * cosx
            bxn *= cosy
            bxn *= cosz

            sinx = np.sinh(kxn*x)
            siny = np.sin(kyn*y)

            byn = (cmat[nx, nz]*kyn/kxn) * sinx
            byn *= siny
            byn *= cosz

            sinz = np.sin(kzn*z + phimat[nx, nz])

            bzn = (cmat[nx, nz]*kzn/kxn) * cosx
            bzn *= siny
            bzn *= sinz

            bx += bxn
            by += bzn
            bz += bzn
    return bx, by, bz


def field_bz_on_axis_dby(kx, kz, cmat, phimat, x, y, z):
    bx = np.zeros(x.shape, dtype=float)
    by = np.zeros(x.shape, dtype=float)
    bz = np.zeros(x.shape, dtype=float)
    for nx in range(cmat.shape[0]):
        for nz in range(1, cmat.shape[1]):
            kxn = nx*kx
            kzn = nz*kz
            kyn = np.sqrt(kxn*kxn + kzn*kzn)

            cosx = np.cos(kxn*x)
            cosy = np.cosh(kyn*y)
            sinz = np.sin(kzn*z + phimat[nx, nz])

            bzn = cmat[nx, nz] * cosx
            bzn *= cosy
            bzn *= cosz

            sinx = np.sinh(kxn*x)
            siny = np.sin(kyn*y)
            cosz = np.cos(kzn*z + phimat[nx, nz])

            byn = (cmat[nx, nz]*kyn/kxn) * sinx
            byn *= siny
            byn *= cosz


            bzn = (cmat[nx, nz]*kzn/kxn) * cosx
            bzn *= siny
            bzn *= sinz

            bx += bxn
            by += bzn
            bz += bzn
    return bx, by, bz
