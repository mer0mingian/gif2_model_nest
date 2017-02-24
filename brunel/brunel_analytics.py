"""
script for testing the analysis and plots from the brunel network
"""




# USE NITIME FOR PLOTTING THE SPECTRUM:
import scipy.signal as sig
import scipy.stats.distributions as dist
import nitime.algorithms as tsa
# import nitime.utils as utils
# from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate

# def dB(x, out=None):
#     if out is None:
#         return 10 * np.log10(x)
#     else:
#         np.log10(x, out)
#         np.multiply(out, 10, out)
# ln2db = dB(np.e)
# freqs, d_psd = tsa.periodogram(timeseries_all)

f, adaptive_psd_mt, nu = tsa.multi_taper_psd(timeseries_all, 
                                adaptive=True, jackknife=False)
# dB(adaptive_psd_mt, adaptive_psd_mt)
# p975 = dist.chi2.ppf(.975, nu)
# p025 = dist.chi2.ppf(.025, nu)
# l1 = ln2db * np.log(nu / p975)
# l2 = ln2db * np.log(nu / p025)
# hyp_limits = (adaptive_psd_mt + l1, adaptive_psd_mt + l2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Power spectrum Layer 5')
ax.plot(f, adaptive_psd_mt, color='k', markersize=3)
ax.set_yscale("Log")
# fig06 = plot_spectral_estimate(freqs, psd, (adaptive_psd_mt,), hyp_limits,
#                        elabels=('MT with adaptive weighting and 95% interval',))
plt.savefig('driven_plots/spectrum_test.png')
