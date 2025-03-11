# See https://stackoverflow.com/questions/51318981/how-to-use-python-to-separate-two-gaussian-curves

from itertools import starmap

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.stats import norm

sns.set(color_codes=True)
# if in jupiter notebook
# %matplotlib inline
# else add this after your plot
# plt.show()


# generate synthetic data from a mixture of two Gaussians with equal weights
# the solution below readily generalises to more components
nsamples = 10000
means = [4*60, 12*60]
sds = [50, 10]
weights = [0.95, 0.05]
draws = np.random.multinomial(nsamples, weights)
samples = np.concatenate(
    list(starmap(np.random.normal, zip(means, sds, draws)))
)
sns.displot(samples, kde=True, color="blue")
plt.show()


from sklearn.mixture import GaussianMixture

mixture = GaussianMixture(n_components=2).fit(samples.reshape(-1, 1))
means_hat = mixture.means_.flatten()
weights_hat = mixture.weights_.flatten()
sds_hat = np.sqrt(mixture.covariances_).flatten()

print(mixture.converged_)
print(means_hat)
print(sds_hat)
print(weights_hat)

mu1_h, sd1_h = means_hat[0], sds_hat[0]
x_axis = np.linspace(mu1_h-3*sd1_h, mu1_h+3*sd1_h, 1000)
plt.plot(x_axis, norm.pdf(x_axis, mu1_h, sd1_h), color="green")


mu2_h, sd2_h = means_hat[1], sds_hat[1]
x_axis_2 = np.linspace(mu2_h-3*sd2_h, mu2_h+3*sd2_h, 1000)
plt.plot(x_axis_2, norm.pdf(x_axis_2, mu2_h, sd2_h), color="red")

plt.show()
