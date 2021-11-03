'''MIT License

Copyright (c) 2021 Roudranil Das

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import arviz as az

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import scipy
from scipy import stats
from scipy.stats.mstats import mquantiles
from scipy.stats import gaussian_kde as gkde

import theano.tensor as tt

import pymc3 as pm

plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.grid'] = True
az.rcParams['stats.hdi_prob'] = 0.95
# %config InlineBackend.figure_format = 'retina' # primary recommended usage is in jupyter notebook
az.style.use(["arviz-darkgrid", "arviz-orangish"])
mpl.style.use('seaborn-whitegrid')
mpl.rcParams['font.size'] = 14
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25


# -----Models start here-----

# Entire group
v_samples_0 = stats.beta.rvs(0.010101+30, 1+5807-30, size=1000000)
c_samples_0 = stats.beta.rvs(0.010101+101, 1+5829-101, size=1000000)
ve_samples_0 = 100*(1-(v_samples_0/c_samples_0))

print("Mean ve: ", ve_samples_0.mean())
print("95% HDI: ", az.stats.hdi(ve_samples_0))
print("95% CI (i guess): ", mquantiles(ve_samples_0, prob=[0.025, 0.975]))

# group 1
v_samples_1 = stats.beta.rvs(0.010101+3, 1+1367-3, size=1000000)
c_samples_1 = stats.beta.rvs(0.010101+30, 1+1374-30, size=1000000)
ve_samples_1 = 100*(1-(v_samples_1/c_samples_1))

print("Mean ve: ", ve_samples_1.mean())
print("95% HDI: ", az.stats.hdi(ve_samples_1))
print("95% CI (i guess): ", mquantiles(ve_samples_1, prob=[0.025, 0.975]))

# group 2
v_samples_2 = stats.beta.rvs(0.010101+15, 1+2377-15, size=1000000)
c_samples_2 = stats.beta.rvs(0.010101+38, 1+2430-38, size=1000000)
ve_samples_2 = 100*(1-(v_samples_2/c_samples_2))

print("Mean ve: ", ve_samples_2.mean())
print("95% HDI: ", az.stats.hdi(ve_samples_2))
print("95% CI (i guess): ", mquantiles(ve_samples_2, prob=[0.025, 0.975]))

# group 3
v_samples_3 = stats.beta.rvs(0.010101+12, 1+2063-12, size=1000000)
c_samples_3 = stats.beta.rvs(0.010101+33, 1+2025-33, size=1000000)
ve_samples_3 = 100*(1-(v_samples_3/c_samples_3))

print("Mean ve: ", ve_samples_3.mean())
print("95% HDI: ", az.stats.hdi(ve_samples_3))
print("95% CI (i guess): ", mquantiles(ve_samples_3, prob=[0.025, 0.975]))


# -----plots start here-----

# For the entire group
# kde plot for virr vs cirr
fig = plt.figure(1, figsize=(12, 6))
plt.axes(frameon=True)

# plotting the data
sns.kdeplot(v_samples_0, color="#FF5733", label="Vaccine IRR", lw=1.5)
sns.kdeplot(c_samples_0, color="#2E86C1", label="Control IRR", lw=1.5)

# ticks
max = 5
xticks = np.arange(0, 0.0275, 0.0025)
yticks = np.arange(0, 100*max+1, 100)

plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=2.5, 
                bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, 
                bottom=True, top=True, left=True, right=True, labelsize=18)
plt.xticks(rotation=45)
plt.xticks(xticks)
plt.yticks(yticks)

# limits
xmin = 0; xmax = 0.025
ymin = 0; ymax = 100*max
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# grid
plt.grid(b=True, which='major', color='k', alpha=1,
        ls=(0, (5, 5)), lw=0.5)
plt.grid(b=True, which='minor', color='k', alpha=0.2,
        ls=(0, (5, 6)), lw=0.5)

# labels and titles
plt.xlabel("Incidence Rate Ratio (IRR) values.", fontsize=20)
plt.ylabel("Posterior density of the sampled values.", fontsize=22)

# legend
plt.legend(frameon=True, fontsize=18, framealpha=0.4, edgecolor="#283747")

# plt.savefig("trial1___virr_vs_cirr__ovr.png", dpi=600)
plt.show();

# kde plot with 95% CI for ve
ve0_kde_man = gkde(ve_samples_0)

ci_0_x = mquantiles(ve_samples_0, prob=[0.025, 0.975])
ci_0_y = ve0_kde_man(mquantiles(ve_samples_0, prob=[0.025, 0.975]))

h = 0.005
x_fill_0 = np.arange(ci_0_x[0]+h, ci_0_x[1], h)
y_fill_0 = ve0_kde_man(x_fill_0)

fig = plt.figure(1, figsize=(12, 6))
plt.axes(frameon=True)

# plotting the data
sns.kdeplot(ve_samples_0, color="#FF5733", label="Vaccine Efficacy", lw=1.5)

# 95% CI
plt.vlines(ci_0_x, ymin=[0, 0], ymax=ci_0_y, color='#AE1F00')
plt.fill_between(x=x_fill_0, y1=y_fill_0, color="#FF9D88", alpha=0.2)
plt.text(67, 0.0206, "95% CI", style='italic', size=18)

# ticks
xticks = np.arange(20, 101, 10)
yticks = np.arange(0, 0.08, 0.01)

plt.minorticks_on()
plt.tick_params(direction='in',which='minor', length=2.5, 
                bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=6, 
                bottom=True, top=True, left=True, right=True, labelsize=18)
plt.xticks(xticks)
plt.yticks(yticks)

# limits
xmin = 20; xmax = 100
ymin = 0; ymax = 0.07
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# grid
plt.grid(b=True, which='major', color='k', alpha=1,
        ls=(0, (5, 5)), lw=0.5)
plt.grid(b=True, which='minor', color='k', alpha=0.2,
        ls=(0, (5, 6)), lw=0.5)

# labels and titles
plt.xlabel("Vaccine Efficacy percentage", fontsize=20)
plt.ylabel("Posterior density of the sampled values", fontsize=22)

# legend
plt.legend(frameon=True, fontsize=18, framealpha=0.4, edgecolor="#283747")

# plt.savefig("trial1___virr_vs_cirr__ovr.png", dpi=600)
plt.show();