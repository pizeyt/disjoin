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
sds = [30, 30]
weights = [0.995, 0.005]
draws = np.random.multinomial(nsamples, weights)
samples = np.concatenate(
    list(starmap(np.random.normal, zip(means, sds, draws)))
)
sns.displot(samples, kde=True, color="blue")
plt.show()
plt.close()


from sklearn.mixture import GaussianMixture

mixture = GaussianMixture(n_components=2).fit(samples.reshape(-1, 1)) # a single feature, use (1,-1) for a single sample
means_hat = mixture.means_.flatten()
weights_hat = mixture.weights_.flatten()
sds_hat = np.sqrt(mixture.covariances_).flatten()

# Work out which is the problem set
if means_hat[0] > means_hat[1]:
    problem = 0
    normal = 1
else:
    problem = 1
    normal = 0

print(mixture.n_components)
print(mixture.converged_)
print(means_hat)
print(sds_hat)
print(weights_hat)

mu_normal, sd_normal = means_hat[normal], sds_hat[normal]
x_axis_normal = np.linspace(mu_normal-3*sd_normal, mu_normal+3*sd_normal, 1000)
plt.plot(x_axis_normal, norm.pdf(x_axis_normal, mu_normal, sd_normal), color="green")


mu_problem, sd_problem = means_hat[problem], sds_hat[problem]
x_axis_problem = np.linspace(mu_problem-3*sd_problem, mu_problem+3*sd_problem, 1000)
plt.plot(x_axis_problem, norm.pdf(x_axis_problem, mu_problem, sd_problem), color="red")

plt.show()
plt.close()

print("Mean of normal set: ", round(means_hat[normal]))
print("Mean of problem set: ", round(means_hat[problem]))
print("Mean of whole set: ", round(np.mean(samples)))

print("Difference of means: ", round((means_hat[problem] - means_hat[normal])))
print("Difference of means as SDs: ", round((means_hat[problem] - means_hat[normal])/sds_hat[normal]))

cutoff = means_hat[normal] + 3 * sds_hat[normal]
good, bad = [], []
for x in samples:
    if x > cutoff:
        bad.append(x)
    else:
        good.append(x)
print(len(good))
print(len(bad))