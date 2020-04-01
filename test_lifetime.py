# %%
import numpy as np

import pyaccel
from pymodels import si
import mathphys

# This code works with mathphys at v1.1.0 and pyaccel at 55feac4

# %%
ring = si.create_accelerator()
ring[651].voltage = 1.5e6
eqpar = pyaccel.optics.EquilibriumParameters(ring)
current = 100  # mA
en_accep = np.array([-1, 1]) * eqpar.rf_acceptance  # * 0.04
energy = eqpar.accelerator.energy / 1e9
n = mathphys.beam_optics.calc_number_of_electrons(
    energy, current/864, ring.length)
coupling = 0.01
avg_pressure = 1e-9

accepx, accepy, *_ = pyaccel.optics.get_transverse_acceptance(
    ring, eqpar.twiss, energy_offset=0.0)
accep_x = min(accepx)
accep_y = min(accepy)
accep_t = [accep_x, accep_y]

lifetime = pyaccel.lifetime.Lifetime(ring)
lifetime.coupling = coupling


# %%
lt = mathphys.beam_lifetime.calc_touschek_loss_rate(
    en_accep, eqpar.twiss, coupling, n, eqpar.emit0, energy,
    eqpar.espread0, eqpar.bunch_length)
ltp = lifetime.lossrate_touschek
lt, ltp


# %%
1 / ltp / 3600


# %%
lq = mathphys.beam_lifetime.calc_quantum_loss_rates(
    accep_t, en_accep[1], coupling,
    natural_emittance=eqpar.emit0, energy_spread=eqpar.espread0,
    damping_times=[eqpar.taux, eqpar.tauy, eqpar.taue])
lqp = lifetime.lossrate_quantum
lq, lqp


# %%
spos = eqpar.twiss.spos
le = mathphys.beam_lifetime.calc_elastic_loss_rate(
    accep_t, avg_pressure, betax=eqpar.twiss.betax,
    betay=eqpar.twiss.betay, energy=energy)
le = np.trapz(le, spos) / (spos[-1]-spos[0])
lep = lifetime.lossrate_elastic
le, lep


# %%
1 / np.mean(lep) / 3600

# %%
li = mathphys.beam_lifetime.calc_inelastic_loss_rate(en_accep[1], avg_pressure)
lip = lifetime.lossrate_inelastic
li, lip

# %%
1 / lip / 3600


# %%
