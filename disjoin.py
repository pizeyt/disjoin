# See https://stackoverflow.com/questions/51318981/how-to-use-python-to-separate-two-gaussian-curves

from itertools import starmap

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from scipy.stats import norm

sns.set(color_codes=True)
# if in jupiter notebook
# %matplotlib inline
# else add this after your plot
# plt.show()

def readCsv():
    print("Data from CSV")
    data = []
    with open('data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(round(float(row['CNV_TIME_HR']), 2))
    return np.array(data)


# generate synthetic data from a mixture of two Gaussians with equal weights
# the solution below readily generalises to more components
def generateData():
    print("Data generated")
    nsamples = 10000
    means = [4*60, 12*60]
    sds = [20, 30]
    weights = [0.995, 0.005]
    draws = np.random.multinomial(nsamples, weights)
    return  np.concatenate(list(starmap(np.random.normal, zip(means, sds, draws))))

def readData():
    print("Data from TXT")
    data = []
    with open('data.txt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = float(line.strip())
            if t>0.00001 and t < 1000 :
                data.append(t)
    return np.array(data)

def filter(data : np.array):
    return sd_filter(data)
def null_filter(data : np.array):
    return data
def sd_filter(data : np.array):
    deviations = 3
    median = np.median(data)
    sd = np.std(data)
    cutoff  = median + sd * deviations
    good, bad = [], []
    for x in data:
        if x > cutoff:
            bad.append(x)
        else:
            good.append(x)
    print ("Ugly:", len(data))
    print ("Good:", len(good))
    print ("Bad:", len(bad))
    print ("Cutoff:", cutoff)
    return np.array(good)



#samples = readCsv()
samples = readData()
#samples = generateData()
print(type(samples))
print("Sample Size:", samples.size)

samples = filter(samples)

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

print("Normal set: ", normal)
print("Problem set: ", problem)
print("Components: ", mixture.n_components)
print(mixture.converged_)
print(means_hat)
print("SDs:", sds_hat)
print("Weights:", weights_hat)

mu_normal, sd_normal = means_hat[normal], sds_hat[normal]
x_axis_normal = np.linspace(mu_normal-2*sd_normal, mu_normal+2*sd_normal, 1000)
plt.plot(x_axis_normal, norm.pdf(x_axis_normal, mu_normal, sd_normal), color="green")


mu_problem, sd_problem = means_hat[problem], sds_hat[problem]
x_axis_problem = np.linspace(mu_problem-2*sd_problem, mu_problem+2*sd_problem, 1000)
plt.plot(x_axis_problem, norm.pdf(x_axis_problem, mu_problem, sd_problem), color="red")

plt.show()
plt.close()

print("Mean of normal set: ", round(means_hat[normal]))
print("Mean of problem set: ", round(means_hat[problem]))
print("Mean of whole set: ", round(np.mean(samples)))


print("Difference of means: ", round((means_hat[problem] - means_hat[normal])))
distance_in_SDs = round((means_hat[problem] - means_hat[normal])/sds_hat[normal])
print("Difference of means in SDs: ", distance_in_SDs)
print("SD Ratios (p/n): ", sds_hat[problem]/sds_hat[normal])



cutoff = means_hat[normal] + (distance_in_SDs/2) * sds_hat[normal]
cutoff=10
print("cutoff: ", cutoff)
good, bad = [], []
for x in samples:
    if x > cutoff:
        bad.append(x)
    else:
        good.append(x)
print("Good:",len(good))
print("Bad:", len(bad))