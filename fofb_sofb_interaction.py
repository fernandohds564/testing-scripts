import numpy as np
from siriuspy.clientconfigdb import ConfigDBClient
import matplotlib.pyplot as mplt

def calc_inverse_matrix(
        mat, bpm_enbl, cor_enbl, min_sv=0.0, regc=0.0, return_svd=False):
    sel_mat = bpm_enbl[:, None] * cor_enbl[None, :]
    mats = mat[sel_mat].reshape(np.sum(bpm_enbl), np.sum(cor_enbl))
    uuu, sing, vvv = np.linalg.svd(mats, full_matrices=False)
    idcs = sing > min_sv
    singr = sing[idcs]

    regc = regc * regc
    inv_s = np.zeros(sing.size, dtype=float)
    inv_s[idcs] = singr/(singr*singr + regc)

    singp = np.zeros(sing.size, dtype=float)
    singp[idcs] = 1/inv_s[idcs]
    imat = np.dot(vvv.T*inv_s, uuu.T)

    inv_mat = np.zeros(mat.shape, dtype=float).T
    inv_mat[sel_mat.T] = imat.ravel()

    if return_svd:
        return inv_mat, (uuu, sing, vvv)
    return inv_mat


def get_filter(s, w0, delay=0, size=320):
    return np.eye(size)/(1+s/w0) * np.exp(-delay*s)


def get_integral_controller(s, mat, gain=1):
    return gain/s * mat


def get_system_matrices(freq):
    s = 2j*np.pi*freq
    Hs = get_filter(s, w0=13*2*np.pi, delay=40e-3, size=320)
    Hf = get_filter(s, w0=13e3*2*np.pi, delay=40e-6, size=320)
    Cs = get_integral_controller(s, mat=imat_s, gain=1*0.5)
    Cf = get_integral_controller(s, mat=imat_f, gain=0*0.00001*25e3)
    Hk = get_filter(s, w0=13*2*np.pi, size=161)
    Dk = get_integral_controller(s, mat=np.dot(imat_s, mat_f), gain=0*0.02*25)
    Dk = Dk @ Hk
    Ou = get_integral_controller(s, mat=mat_s, gain=0.0*25)

    Gs = mat_s @ get_filter(s, w0=1e2*2*np.pi, delay=40e-3, size=281) 
    Gf = mat_f @ get_filter(s, w0=1e4*2*np.pi, delay=40e-6, size=161)

    T = Gs @ Cs @ Hs + (Gs @ Dk + Gf) @ Cf @ (Hf + Ou @ Cs @ Hs)

    Td2y = np.linalg.pinv(np.eye(320) + T)
    Td2kf = - Cf @ (Hf + Ou @ Cs @ Hs) @ Td2y
    Td2ks = - Cs @ Hs @ Td2y - Dk @ Td2kf
    return Td2y, Td2kf, Td2ks


clt = ConfigDBClient()

mat_s = np.array(clt.get_config_value(
    'ref_respmat', config_type='si_orbcorr_respm'))
mat_f = np.array(clt.get_config_value(
    'ref_respmat', config_type='si_fastorbcorr_respm'))

bpmxenbl_f = np.ones(160, dtype=bool).reshape(-1, 8)
bpmyenbl_f = np.ones(160, dtype=bool).reshape(-1, 8)
bpmxenbl_f[:, [1, 2, 3, 4, 5, 6]] = False
bpmyenbl_f[:, [1, 2, 3, 4, 5, 6]] = False
bpm_enbl_f = np.r_[bpmxenbl_f.ravel(), bpmyenbl_f.ravel()]

chenbl_f = np.ones(80, dtype=bool).reshape(-1, 4)
cvenbl_f = np.ones(80, dtype=bool).reshape(-1, 4)
chenbl_f[:, [1, 2, 3]] = False
cvenbl_f[:, [1, 2, 3]] = False
rfenbl_f = False
cor_enbl_f = np.r_[chenbl_f.ravel(), cvenbl_f.ravel(), rfenbl_f]

bpmxenbl_s = np.ones(160, dtype=bool).reshape(-1, 8)
bpmyenbl_s = np.ones(160, dtype=bool).reshape(-1, 8)
bpm_enbl_s = np.r_[bpmxenbl_s.ravel(), bpmyenbl_s.ravel()]

chenbl_s = np.ones(120, dtype=bool).reshape(-1, 6)
cvenbl_s = np.ones(160, dtype=bool).reshape(-1, 8)
rfenbl_s = True
cor_enbl_s = np.r_[chenbl_s.ravel(), cvenbl_s.ravel(), rfenbl_f]

imat_s = calc_inverse_matrix(
    mat_s, bpm_enbl_s, cor_enbl_s, min_sv=0.0, regc=0.0, return_svd=False)
imat_f = calc_inverse_matrix(
    mat_f, bpm_enbl_f, cor_enbl_f, min_sv=0.0, regc=0.0, return_svd=False)


freqs = np.linspace(1e-1, 1e2, 500)
sd2ys = []
for freq in freqs:
    Td2y, Td2kf, Td2ks = get_system_matrices(freq)
    _, sd2y, _ = np.linalg.svd(Td2y, full_matrices=False)
    sd2ys.append(sd2y)

sd2ys = np.array(sd2ys)
mplt.plot(freqs, sd2ys)
mplt.xscale('log')
mplt.yscale('log')
mplt.show()




