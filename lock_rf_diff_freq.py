# from numpy import round, abs
import numpy as np

# Lock injection with to different RF frequencies

rfbo = 499661000
prec = 2
tdrift = 100  # in hours

maxdeltaf = 5000
mindeltaf = 500

maxdiv = rfbo/mindeltaf/4
maxdiv_corr = int(maxdiv)
mindiv = rfbo/maxdeltaf/4
mindiv_corr = int(mindiv)


ns = np.arange(mindiv_corr, maxdiv_corr)

freq_diff = rfbo/4/ns
rfsi_corr = rfbo + freq_diff

rfsi_round = np.round(rfsi_corr, decimals=prec)
rfsi_diff = rfsi_corr-rfsi_round

period = 1/np.abs(rfsi_diff)/3600

ind = np.logical_and(rfsi_diff != 0, period < tdrift)
nind = np.logical_not(ind)

freq_diff = freq_diff[nind]
rfsi_round = rfsi_round[nind]
ns = ns[nind]
period = period[nind]

ftmpl = '%16.{:d}f'.format(prec)
ftmplp = '{{:0.{:d}f}}'.format(prec)
arr = np.vstack([freq_diff, rfsi_round, ns, period]).T
np.savetxt(
    'data.txt', arr, fmt=[ftmpl, ftmpl, '%10d', '%20.1f'],
    header='{0:>16s} {1:>16s} {2:>10s} {3:>20s}'.format(
        'Delta freq [Hz]', 'SI Freq [Hz]', 'Clk7 Div', 'Dephasing Time [h]'),
    comments=(
        ('# Minimum delta freq [Hz]: '+ftmplp+' [Hz]\n').format(mindeltaf) +
        ('# Maximum delta freq [Hz]: '+ftmplp+' [Hz]\n').format(maxdeltaf) +
        '# Mininum dephasing Time: {0:.1f} [h]\n'.format(tdrift) +
        ('# Booster RF Frequency: '+ftmplp+' [Hz]\n').format(rfbo) +
        '# Frequency Precision : {0:d} decimals \n\n'.format(prec))
    )
