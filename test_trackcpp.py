from pymodels import si
import pyaccel

mod = si.create_accelerator()
mod.cavity_on = True
mod.radiation_on = True
orb = pyaccel.tracking.find_orbit6(mod)
print('orb1', orb)

ind = pyaccel.lattice.find_indices(mod, 'pass_method', 'cavity_pass')

freq = mod[ind[0]].frequency
freq_new = freq + 2e3
freq_rev = mod.velocity/mod.length / mod.beta_factor
hnum = freq_new / freq_rev

mod[ind[0]].frequency = freq_new
mod.harmonic_number = hnum

orb = pyaccel.tracking.find_orbit6(mod)
print('orb2', orb)
