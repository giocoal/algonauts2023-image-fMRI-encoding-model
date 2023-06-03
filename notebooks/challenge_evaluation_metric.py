import numpy as np

# "Xh_correlation" are 1-D vectors with the correlation scores of all vertices
# of a given Challenge subject (each component corresponds to the correlation
# score of a vertex).

# "Xh_noise_ceiling" are 1-D vectors with the noise ceiling values of all
# vertices of a given Challenge subject (each component corresponds to the noise
# ceiling value of a vertex).

# Set negative correlation values to 0, so to keep the noise-normalized
# encoding accuracy positive
lh_correlation[lh_correlation<0] = 0
rh_correlation[rh_correlation<0] = 0
# Square the correlation values
lh_correlation = lh_correlation ** 2
rh_correlation = rh_correlation ** 2
# Add a very small number to noise ceiling values of 0, otherwise the
# noise-normalized encoding accuracy cannot be calculated (division by 0 is
# not possible)
lh_noise_ceiling[lh_noise_ceiling==0] = 1e-14
rh_noise_ceiling[rh_noise_ceiling==0] = 1e-14
# Compute the noise-normalized encoding accuracy
lh_noise_norm_corr = np.divide(lh_correlation, lh_noise_ceiling)
rh_noise_norm_corr = np.divide(rh_correlation, rh_noise_ceiling)
# Set the noise-normalized encoding accuracy to 1 (100% accuracy) for those
# vertices in which the correlation is higher than the noise ceiling, to prevent
# encoding accuracy values higher than 100%
lh_noise_norm_corr[lh_noise_norm_corr>1] = 1
rh_noise_norm_corr[rh_noise_norm_corr>1] = 1
