from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from numpy.polynomial.polynomial import polyfit
import numpy as np
from scipy.stats import binom_test
from scipy.stats import binom
import seaborn as sns
import datetime
from scipy import optimize



# set up the ground truth
sample_size = 100000
expected_value = lambda_ = 4.5
poi = np.random.poisson
N_samples = range(1, sample_size, 100)

for k in range(3):
    samples = poi(lambda_, sample_size)

    partial_average = [samples[:i].mean() for i in N_samples]

    plt.plot(N_samples, partial_average, lw=1.5, label="average \
of  $n$ samples; seq. %d" % k)

plt.plot(N_samples, expected_value * np.ones_like(partial_average),
         ls="--", label="true expected value", c="k")

plt.ylim(4.35, 4.65)
plt.title("Convergence of the average of \n random variables to its \
expected value")
plt.ylabel("average of $n$ samples")
plt.xlabel("# of samples, $n$")
plt.legend()
plt.show()

import random
normal = np.random.normal(0, 1, 1000)
poisson = np.random.poisson(4.5, 1000)
uniform = np.random.uniform(-4, 4, 1000)
binomial = np.random.binomial(10, .8, 1000)

bins = np.linspace(-5, 10, 100)

plt.hist(normal, bins, alpha=0.5, label='normal(0, 1)')
plt.hist(poisson, bins, alpha=0.5, label='poisson(4.5)')
plt.hist(uniform, bins, label='uniform(-4, 4)')
plt.hist(binomial, bins, label='binomial(10, .8)')
plt.legend(loc='upper left')
plt.title("Variety of distributions with parameter specifcations")
plt.xlabel("Realisation of Variable ")
plt.ylabel("Frequency of Observation")
plt.show()



### Piecewise Linear Fits

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  dtype=float)
y = np.array([5, 12, 9, 11, 13, 10,
              28.92, 42.81, 56.7, 70.59, 84.47, 75.36, 112.25, 100.14, 140.03, 3, 70, 300, 5, 100, 50], dtype=float)

def piecewise_linear(x, b1, a1,  b2, a2, b3, a3):
    funcs = [lambda x:b1*x + a1,
             lambda x:b2*x + a2,
             lambda x:b3*x + a3]
    conds = [x < 7, ((x >= 7) & (x < 15)), x > 15]
    return np.piecewise(x, conds, funcs)

p , e = optimize.curve_fit(piecewise_linear, x, y, method="trf")
a, b = polyfit(x, y, 1)
xd = np.linspace(0, 20, 1000)

plt.plot(x, y, "o")
plt.plot(x, a + b * x, '--', label="Global fit: {b: .2f}x + {a: .2f}".format(b=b, a=a))
plt.plot(xd, piecewise_linear(xd, *p), label="Linear fit 1: {b: .2f}x + {a: .2f} \n"
                                             "Linear fit 2: {b1: .2f}x + {a1: .2f} \n"
                                             "Linear fit 3: {b2: .2f}x + {a2: .2f}".format(b=p[0], a=p[1],
                                                                                          b1=p[2], a1=p[3],
                                                                                          b2=p[4], a2=p[5]))
plt.title("Piecewise Linear Fits of Series at Changepoints")
plt.legend()
plt.show()

#### Sampling Distributions of Linear Fits
#### Build True Model
N = 100000
X = np.random.uniform(0, 20, N)
uncorrelated_errors = np.random.normal(0, 10, N)
correlated_errors = np.random.uniform (0, 10) + np.sin(np.linspace(0, 10*np.pi, N)) \
          + np.sin(np.linspace(0, 5*np.pi, N))**2 \
          + np.sin(np.linspace(1, 6*np.pi, N))**2

Y_corr = -2 + 3.5 * X + correlated_errors
Y = -2 + 3.5 * X + uncorrelated_errors
population = pd.DataFrame({'X': X, 'Y': Y, 'Y_corr': Y_corr})

fits = pd.DataFrame(columns=['iid_const', 'iid_beta', 'corr_const', 'corr_beta'])
for i in range(0, 10000):
    sample = population.sample(n=100, replace=True)
    Y = sample['Y']; X = sample['X'] ; Y_corr = sample['Y_corr']
    X = sm.add_constant(X)
    iid_model = sm.OLS(Y, X)
    results = iid_model.fit()
    corr_model = sm.OLS(Y_corr, X)
    results_2 = corr_model.fit()
    row = [results.params[0], results.params[1], results_2.params[0], results_2.params[1]]
    fits.loc[len(fits)] = row


fits.boxplot()
plt.suptitle("The Sampling Distribution of Parameters for a Linear models")
plt.title("Based on 10,000 fits on 100 observations")
plt.show()


def draw_gaussian_at(position, sample, ax_main=None, model='iid', color='k', **kwargs):
    filter_var = round(sample['X'], 0) == position
    avg = sample[filter_var]['predicted_Y_' + model].mean()
    min = sample[filter_var]['Y'].min()
    max = sample[filter_var]['Y'].max()
    dist = pd.Series(sample[filter_var]['predicted_error_'+ model].values)
    kde = sm.nonparametric.KDEUnivariate(dist)
    kde.fit()
    density = kde.density
    density /= density.max()
    density *= 1
    y_axis = np.linspace(min, max, len(density))
    label = "Expected error X = {x:}: {err: .2f}".format(err = sample[filter_var]['predicted_error_'+ model].mean(),
                                                         x=position)
    ax_main.plot((density + position), y_axis, color=color, label=label)

sample = population.sample(n=1000, replace=True, random_state=100)
true_model = sm.OLS(sample['Y'], sample['X']).fit()
error_model = sm.OLS(sample['Y_corr'], sample['X']).fit()
sample['predicted_Y_iid'] = true_model.predict(sample['X'])
sample['predicted_Y_corr'] = error_model.predict(sample['X'])
sample['predicted_error_iid'] = sample['Y'] - sample['predicted_Y_iid']
sample['predicted_error_corr'] = sample['Y'] - sample['predicted_Y_corr']
fig, ax1 = plt.subplots()
ax1.plot(sample['X'], sample['Y'],'o')
ax1.plot(sample['X'], sample['predicted_Y_iid'])
for each in [5, 10,  15]:
    d = draw_gaussian_at(position=each, sample=sample, ax_main=ax1, model='iid', color='r')
plt.show()
plt.title("Error Distributions around predicted Y values for the {model:}".format(model='iid model'))
plt.legend()





### Basic t-test

## Define 2 random distributions

#Gaussian distributed data with mean = 2 and var = 1
mean_100 = np.random.normal(100, 20, 1000)

#### Set up p-value Plot

samples = 5
prop = 0.4
successes = 3

X = stats.binom(samples, prop)
x = X.rvs(1000)
points = sns.distplot(x, hist=False, kde=True).get_lines()[0].get_data()

prob = 1 - X.cdf(2)

z = points[0]
y = points[1]
plt.fill_between(z,y, where = z >= successes, color='r', label="Probability >= 3:{p: .2f}".format(p=prob))
plt.fill_between(z,y, where = z < successes , color='g',
                 label="Probability < 3: {mean: .2f}".format(mean=1-prob))
plt.legend(loc='upper right', title='Legend')
plt.title("The P-Value for a sample of 5 with >=3 head \n given a biased coin with expected proportion 0.4")
plt.xlim((0, 5))
plt.yticks([])
plt.show()

result = binom_test(successes, samples, prop, "greater")
print(result)

prior = 0.5
h0 = stats.binom(5, 0.4).pmf(3)
h1 = stats.binom(5, 0.5).pmf(3)

ph0 = (prior*h0) / (prior*h0 + prior*h1)
ph1 = (prior*h1) / (prior*h0 + prior*h1)

print(ph0, ph1, ph1/ph0)

print(binom_test(3, 5, .4, "greater"))
print(binom_test(3, 5, .5, "greater"))

#### Conjugate Priors

beta_params = [(.98, .02), (0.01, 0.01), (.02, .10), (0.01, 1)]
x = np.linspace(0.01, .99, 100)
beta = stats.beta
for a, b in beta_params:
    y = beta.pdf(x, a,  b)
    lines = plt.plot(x, y, label = "params: {a: .2f} , {b: .2f}".format(a=a, b=b))
    #lines = plt.plot(x, y_norm, label="params: {a: .2f} , {b: .2f}".format(a=a, b=b))
    plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_color())
    plt.autoscale(tight=True)

plt.ylim(0)
plt.legend(loc="upper left", title="(a, b) - parameters")
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Beta Distribution")

y_norm = np.random.normal(0.5, 0.2, 100000)
#y_norm = [1 if x >= .75 else 0 if x < .1 else x for x in y_norm]
beta_transform = beta.pdf(y_norm, 0.75, 0.01)
#beta_transform = stats.logistic.pdf(y_norm)
df = pd.DataFrame({'y_norm': y_norm, 'beta_transform': beta_transform})
df.sort_values('y_norm', inplace=True)
df['beta_transform'] = [1 if x > 1 else x for x in df['beta_transform']]
plt.plot(df['y_norm'], df['beta_transform'])
plt.xlim((0, 1))
plt.ylim((0, 1))


### Multinomial distribution

m_var = stats.multinomial(n=100, p=[.3, .4, .2, .1])
m_var_sample = m_var.rvs(1000)
m_var_2 = stats.multinomial(n=100, p=[.3, .4, .1, .2])
m_var_2_sample = m_var_2.rvs(20)
base = datetime.datetime.today() - datetime.timedelta(days=1000)
df = pd.DataFrame(m_var_sample, columns=['plan_1', 'plan_2', 'plan_3', 'no_plan'])
df = df.append(pd.DataFrame(m_var_2_sample, columns=['plan_1', 'plan_2', 'plan_3', 'no_plan']),
               ignore_index=True)
df['totals'] = df['plan_1'] + df['plan_2'] + df['plan_3'] + df['no_plan']
df['plan_1_rate'] = df['plan_1'] / df['totals']
df['plan_2_rate'] = df['plan_2'] / df['totals']
df['plan_3_rate'] = df['plan_3'] / df['totals']
df['no_plan_rate'] = df['no_plan'] / df['totals']
date_list = [base + datetime.timedelta(days=x) for x in range(len(df))]
df.index = date_list
for i, column in enumerate([x for x in df.columns if 'rate' in x]):
    ax = plt.plot(df[column], label=column)

plt.title("Outcomes of Signup Process after Website Change")
plt.legend(loc="upper left")
plt.xticks(rotation=45)
plt.show()

#### Calculate Expected Revenue

def expected_revenue(posterior_samples):
    return 10*posterior_samples[:, 0] + 7*posterior_samples[:, 1] + \
           12*posterior_samples[:, 2] + 0*posterior_samples[:, 3]

full_data = df[['plan_1', 'plan_2', 'plan_3', 'no_plan']]
weird_data2 = df[['plan_1', 'plan_2', 'plan_3', 'no_plan']].tail(20)
normal_data = df[['plan_1', 'plan_2', 'plan_3', 'no_plan']].head(1000)

# Bayesian Posterior
multinomial_posterior_new = np.random.dirichlet(np.array([100, 100, 100, 100]) +
                                                 np.array(weird_data2.sum()), size=1000)

# Empirical Bayes
multinomial_posterior_full = np.random.dirichlet(np.array([100, 100, 100, 100]) + np.array(full_data.sum()), size=1000)

# Bayesain Posterior Prior Data

multinomial_posterior_old = np.random.dirichlet(np.array([100, 100, 100, 100]) + np.array(normal_data.sum()), size=1000)

multinomial_posterior_new_empirical_prior = np.random.dirichlet(np.array([300, 400, 200, 100]) +
                                                 np.array(weird_data2.sum()), size=1000)

multinomial_posterior_crazy_prior = np.random.dirichlet(np.array([100, 400, 500, 0]) +
                                                 np.array(weird_data2.sum()), size=1000)


expected_value_new = expected_revenue(multinomial_posterior_new)
expected_value_old = expected_revenue(multinomial_posterior_old)
expected_value_full_data = expected_revenue(multinomial_posterior_full)
expected_value_new_bias = expected_revenue(multinomial_posterior_new_empirical_prior)
expected_value_crazy = expected_revenue(multinomial_posterior_crazy_prior)

plt.hist(expected_value_new, histtype='stepfilled', label="Expected Revenue New (20) Obs Flat Prior", bins=50)
plt.hist(expected_value_old, histtype='stepfilled', label="Expected Revenue Old Obs Flat Prior", bins=50, alpha=0.8)
plt.hist(expected_value_full_data, histtype='stepfilled', label="Expected Revenue Full Obs Flat Prior", bins=50, alpha=0.8)
plt.hist(expected_value_new_bias, histtype='stepfilled', label="Expected Revenue New (20) Obs Empirical Prior ", bins=50, alpha=0.8)
plt.hist(expected_value_crazy, histtype='stepfilled', label="Expected Revenue New (20) Obs Optimistic Prior ", bins=50, alpha=0.8)
plt.title("Expected Revenue of Multinomial Posterior given Data")
plt.legend(loc="upper left")
plt.show()

