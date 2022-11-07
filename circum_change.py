import numpy as np
import scipy.integrate as scyint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# ######## Model with a vertical orbit change. ###########

# circum = 518.4
# amp = 10e-6
# harm = 2
# rad = circum / (2*np.pi)


# def get_coords(par):
#     x = rad*np.cos(par)
#     y = rad*np.sin(par)
#     z = amp*np.cos(harm*par)
#     return x, y, z


# def get_traj_len(par):
#     dx = -rad*np.sin(par)
#     dy = rad*np.cos(par)
#     dz = -harm*amp*np.sin(harm*par)
#     return np.sqrt(dx*dx + dy*dy + dz*dz)


# var_num, err = scyint.quad(get_traj_len, 0, 2*np.pi, epsabs=1e-13)
# var_num -= circum

# var_anal = circum*(harm*amp/rad)**2/4

# print('Circumference Variation:')
# print(f'Analytic: {var_anal*1e6:.2g} um')
# print(f'Numeric: {var_num*1e6:.2g} um')

# phi = np.linspace(0, 1, 10000) * 2*np.pi
# x, y, z = get_coords(phi)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(x, y, z)
# plt.show()


# ####### Model with a localized radius change. #########

circum = 518.4
amp = 100e-6
harm = 9
rad0 = circum / (2*np.pi)


def get_radius(par):
    rad = rad0 + amp*np.sin(par/2)**(2*harm)
    drad = amp * 2*harm * np.sin(par/2)**(2*harm-1)*np.cos(par/2)/2
    return rad, drad


def get_coords(par):
    rad, _ = get_radius(par)
    x = rad*np.cos(par)
    y = rad*np.sin(par)
    return x, y, 0


def get_traj_len(par):
    rad, drad = get_radius(par)
    dx = drad*np.cos(par) - rad*np.sin(par)
    dy = drad*np.sin(par) + rad*np.cos(par)
    dz = 0
    return np.sqrt(dx*dx + dy*dy + dz*dz)


ang = 2*np.arcsin(20/rad0)
x0, y0, z0 = get_coords(np.pi)
x1, y1, z1 = get_coords(np.pi+ang)
dis = np.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0))
print(dis, dis-40)

var_num, err = scyint.quad(get_traj_len, 0, 2*np.pi, epsabs=1e-13)
var_num -= circum

print('Circumference Variation:')
print(f'Numeric: {var_num*1e6:.2f} um')

phi = np.linspace(0, 1, 10000) * 2*np.pi
rad, drad = get_radius(phi)
fig = plt.figure()
ax = fig.gca()
ax.plot(phi, (rad - rad0)*1e6, color='tab:blue')
ay = ax.twinx()
ay.plot(phi, drad*1e6, color='tab:red')
ax.set_title(
    r'$\delta = $' + f'{amp*1e6:.1f} ' + r'$\mu$m, ' +
    f'n = {harm:d} ' + r'$\rightarrow \,\, \delta C = $' +
    f'{var_num*1e6:.2f} ' + r'$\mu$m')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$r-r_0\,\,[\mu m]$')
ay.set_ylabel(r'$\mathrm{d}r/\mathrm{d}\theta\,\,[\mu m]$', color='tab:red')
plt.setp(ay.get_yticklabels(), color='tab:red')
plt.setp(ax.get_yticklabels(), color='tab:blue')
fig.tight_layout()
plt.show()
