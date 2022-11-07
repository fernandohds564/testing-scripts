

import time
from math import sin as _sin, cos as _cos, tan as _tan, \
    sqrt as _sqrt

import numpy as np
import scipy.linalg as scylin

import pyaccel as pyacc
import trackcpp as _trackcpp


# Fourth order-symplectic integrator constants
DRIFT1 = 0.6756035959798286638
DRIFT2 = -0.1756035959798286639
KICK1 = 1.351207191959657328
KICK2 = -1.702414383919314656

# Physical constants used in the calculations
TWOPI = 6.28318530717959
CGAMMA = 8.846056192e-05  # [m]/[GeV^3] Ref[1] (4.1)
M0C2 = 5.10999060e5  # Electron rest mass [eV]
LAMBDABAR = 3.86159323e-13  # Compton wavelength/2pi [m]
CER = 2.81794092e-15  # Classical electron radius [m]
CU = 1.323094366892892  # 55/(24*sqrt(3)) factor


def sandwich_matrix(MKICK, BDIFF):
    """."""
    # return MKICK @ BDIFF @ MKICK.swapaxes(-1, -2)
    return np.dot(MKICK, np.dot(BDIFF, MKICK.T))


def b2_perp(bx, by, irho, x, xpr, y, ypr):
    """."""
    xirho = 1 + x * irho
    b2perp = by*xirho * by*xirho
    b2perp += bx*xirho * bx*xirho
    b2perp += (bx*ypr - by*xpr) * (bx*ypr - by*xpr)
    b2perp /= xirho*xirho + xpr*xpr + ypr*ypr
    return b2perp


# NOTE: I don't understand this pass method.
# There is a discrepancy between edge_fringe from the dipole
# pass method and this one.
# I decided to keep this unchanged so far, to reproce AT results.
# Besides, I don't understand the linearized matriz used to propagate B.
def edgefringe_b(r, diffmat, irho, edge_angle, fint, gap):
    """."""
    if irho <= 0:
        return r, diffmat

    fx = fy = irho * _tan(edge_angle)  # 1/(1+r[4]) ???
    psi = 0
    if fint > 0 and gap > 0:
        psi = irho*gap*fint*(1 + _sin(edge_angle)**2)/_cos(edge_angle)
        fy = irho*_tan(edge_angle - psi/(1+r[4]))  # 1/(1+r[4]) ???
    fac = r[2]*(irho*irho + fy*fy)*psi/(1+r[4])**2/irho

    #  Propagate B
    diffmat[1, :] += fx * diffmat[0, :]
    diffmat[3, :] -= fy * diffmat[2, :]
    if fint > 0 and gap > 0:
        diffmat[3, :] -= diffmat[4, :] * fac
    diffmat[:, 1] += fx * diffmat[:, 0]
    diffmat[:, 3] -= fy * diffmat[:, 2]
    if fint > 0 and gap > 0:
        diffmat[:, 3] -= diffmat[:, 4] * fac

    # Propagate particle
    r[1] += r[0] * fx
    r[3] -= r[2] * fy

    return r, diffmat


def drift_propagate(rin, b_diff, leng):
    """Propagate cumulative Ohmi's diffusion matrix B through a drift."""
    drift = np.eye(6)
    ptot = 1 + rin[4]
    lop = leng/ptot

    # This matrix is the linearization of the drift map.
    # If you derive the map in relation to the coordinates
    # you will get this matrix
    drift[0, 1] = lop
    drift[2, 3] = lop
    drift[0, 4] = -lop*rin[1]/ptot
    drift[2, 4] = -lop*rin[3]/ptot
    drift[5, 1] = lop*rin[1]/ptot
    drift[5, 3] = lop*rin[3]/ptot
    drift[5, 4] = -lop * (rin[1]*rin[1] + rin[3]*rin[3]) / (ptot*ptot)

    rin[0] += lop * rin[1]
    rin[2] += lop * rin[3]
    rin[5] += lop / ptot * (rin[1]*rin[1]+rin[3]*rin[3]) / 2

    return rin, sandwich_matrix(drift, b_diff)


def thinkick_m(rin, pol_a, pol_b, frac, irho, max_order):
    """."""
    im_sum = max_order * pol_a[max_order]
    re_sum = max_order * pol_b[max_order]

    # Recursively calculate the derivatives
    #   ReSumN = (irho/B0)*Re(d(By + iBx)/dx)
    #   ImSumN = (irho/B0)*Im(d(By + iBx)/dy)
    for i in range(max_order-1, 0, -1):
        re_tmp = re_sum*rin[0] - im_sum*rin[2] + i*pol_b[i]
        im_sum = im_sum*rin[0] + re_sum*rin[2] + i*pol_a[i]
        re_sum = re_tmp

    m66 = np.eye(6)
    m66[1, 0] = -frac * re_sum
    m66[1, 2] = frac * im_sum
    m66[3, 0] = frac * im_sum
    m66[3, 2] = frac * re_sum
    m66[1, 4] = frac * irho
    m66[1, 0] += -frac * irho * irho
    m66[5, 0] = frac * irho
    return m66


def thinkick_b(rin, pol_a, pol_b, frac, irho, max_order, E0):
    """Calculate Ohmi's diffusion matrix of a thin multipole element."""
    CRAD = CGAMMA * E0 * E0 * E0 / (TWOPI*1e27)
    p_norm = (1+rin[4])
    p_norm2 = p_norm*p_norm
    im_sum = pol_a[max_order]
    re_sum = pol_b[max_order]

    # calculate angles from momenta
    p_norm = 1 + rin[4]
    x = rin[0]
    y = rin[2]
    xpr = rin[1]/p_norm
    ypr = rin[3]/p_norm
    # save a copy of the initial value of dp/p
    dp_0 = rin[4]

    # recursively calculate the local transvrese magnetic field
    # re_sum = irho*By/B0
    # im_sum = irho*Bx/B0
    for i in range(max_order-1, -1, -1):
        re_tmp = re_sum*rin[0] - im_sum*rin[2] + pol_b[i]
        im_sum = im_sum*rin[0] + re_sum*rin[2] + pol_a[i]
        re_sum = re_tmp

    # calculate |B x n|^3 - the third power of the B field component
    # orthogonal to the normalized velocity vector n
    b2p = b2_perp(im_sum, re_sum + irho, irho, x, xpr, y, ypr)
    b3p = b2p*_sqrt(b2p)

    bb_ = CU*CER*LAMBDABAR * (E0/M0C2)**5 * frac * b3p * p_norm2*p_norm2
    bb_ *= 1 + rin[0]*irho + (rin[1]*rin[1] + rin[3]*rin[3]) / p_norm2 / 2

    # Populate b66
    b66 = np.zeros((6, 6))
    b66[1, 1] = bb_ * rin[1] * rin[1] / p_norm2
    b66[1, 3] = bb_ * rin[1] * rin[3] / p_norm2
    b66[1, 4] = bb_ * rin[1] / p_norm
    b66[3, 3] = bb_ * rin[3] * rin[3] / p_norm2
    b66[3, 4] = bb_ * rin[3] / p_norm
    b66[3, 1] = b66[1, 3]
    b66[4, 1] = b66[1, 4]
    b66[4, 3] = b66[3, 4]
    b66[4, 4] = bb_

    # Update trajectory
    rin[4] -= CRAD*frac*p_norm2*b2p*(1 + x*irho + (xpr*xpr + ypr*ypr)/2)

    # recalculate momenta from angles after losing energy to radiation
    p_norm = 1 + rin[4]
    rin[1] = xpr * p_norm
    rin[3] = ypr * p_norm

    rin[1] -= frac * (re_sum - (dp_0 - rin[0]*irho) * irho)
    rin[3] += frac * im_sum
    rin[5] += frac * irho * rin[0]

    return b66, rin


def propagate_b_diff(ele, rin, energy, b_diff):
    """Find Ohmi's diffusion matrix b_diff of a thick multipole."""
    irho = ele.angle / ele.length
    pola = ele.polynom_a
    polb = ele.polynom_b
    mord = min(pola.size, polb.size) - 1

    # 4-th order symplectic integrator constants
    frac = ele.length / ele.nr_steps
    len1 = frac * DRIFT1
    len2 = frac * DRIFT2
    kick1 = frac * KICK1
    kick2 = frac * KICK2

    # Transform rin to a local coordinate system of the element
    rin += ele.t_in
    rin = np.dot(ele.r_in, rin)

    # Propagate rin and b_diff through the entrance edge
    edgefringe_b(rin, b_diff, irho, ele.angle_in, ele.fint_in, ele.gap)

    # Propagate rin and b_diff through a 4-th order integrator
    for _ in range(ele.nr_steps):
        rin, b_diff = drift_propagate(rin, b_diff, len1)
        for len_, kick in zip((len2, len2, len1), (kick1, kick2, kick1)):
            # Calculate the symplectic transfer map and propagate b_diff:
            m66 = thinkick_m(rin, pola, polb, kick, irho, mord)
            b_diff = sandwich_matrix(m66, b_diff)

            # Propagate rin with radiation terms and calculate element b_diff:
            b66, rin = thinkick_b(rin, pola, polb, kick, irho, mord, energy)
            b_diff += b66

            rin, b_diff = drift_propagate(rin, b_diff, len_)

    edgefringe_b(rin, b_diff, irho, ele.angle_out, ele.fint_out, ele.gap)

    # Transform orbit to the global coordinate system
    rin = np.dot(ele.r_out, rin)
    rin += ele.t_out

    return b_diff


def calc_ohmienvelope(model):
    """."""
    cav_on = model.cavity_on
    rad_on = model.radiation_on
    model.cavity_on = True
    model.radiation_on = True

    orbs = pyacc.tracking.find_orbit6(model, indices='open')
    m66, cummats = pyacc.tracking.find_m66(
        model, indices='closed', closed_orbit=orbs[:, 0])

    mateles = np.linalg.solve(
        cummats[:-1].transpose((0, 2, 1)), cummats[1:].transpose((0, 2, 1)))
    mateles = mateles.transpose((0, 2, 1))

    tini = time.time()
    bdiffs = np.zeros((len(model)+1, 6, 6), dtype=float)
    ene = model.energy
    for i, ele in enumerate(model):
        if ele.pass_method.endswith('_mpole_symplectic4_pass'):
            bdiffs[i+1] = propagate_b_diff(ele, orbs[:, i], ene, bdiffs[i])
        else:
            bdiffs[i+1] = sandwich_matrix(mateles[i], bdiffs[i])
    print(time.time()-tini)

    # ------------------------------------------------------------------------
    # Equation for the moment matrix env is
    #        env = m66 @ env @ m66' + bcum;
    # We rewrite it in the form of the Sylvester equation:
    #        m66i @ env + env @ m66t = bcumi
    # where
    #        m66i =  inv(m66)
    #        m66t = -m66'
    #        bcumi = -m66i @ bcum
    # -----------------------------------------------------------------------
    m66i = np.linalg.inv(m66)
    m66t = -m66.T
    bcumi = np.linalg.solve(m66, bdiffs[-1])

    # Envelope matrix at the ring entrance
    env = scylin.solve_sylvester(m66i, m66t, bcumi)

    rms_dp = np.sqrt(env[4, 4])  # R.M.S. energy spread
    rms_bl = np.sqrt(env[5, 5])  # R.M.S. bunch lenght

    env_list = np.zeros((len(model)+1, 6, 6), dtype=float)
    env_list[0] = env
    for i in range(len(model)):
        env_list[i+1] = sandwich_matrix(cummats[i+1], env) + bdiffs[i]

    model.cavity_on = cav_on
    model.radiation_on = rad_on

    return env_list
