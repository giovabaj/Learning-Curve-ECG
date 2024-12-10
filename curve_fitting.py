from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from uncertainties import ufloat


def func(x, a, b, c):
    return a * x**(-b) + c


# def func(x, a, b, c):
#     return a * np.exp(-b*x) + c


data = pd.read_csv("results/sample1000_100reps.csv")
auc_mean = data.mean().to_numpy()
auc_std = data.std().to_numpy() * 1.96 / np.sqrt(data.shape[0])
n = data.columns.values.astype(int)

popt, pcov = curve_fit(func, n, auc_mean, sigma=auc_std, absolute_sigma=True,  bounds=([-50., -1., 0], [0., 1., .99]))
param_err = np.sqrt(np.diagonal(pcov))

y_pred = func(n, *popt)
r2 = r2_score(auc_mean, y_pred)

r = auc_mean - y_pred
df = len(auc_mean) - 3
chisq = np.sum((r/auc_std)**2) / df
print("reduced chisq =", chisq, " , df =", df)
print("R2 score:", r2)

plt.figure()
plt.errorbar(n, auc_mean, auc_std, None, 'b.', label='data')
n_dense = np.linspace(n[0], n[-1]+100, 100)
plt.plot(n_dense, func(n_dense, *popt), 'r-')
plt.text(450, 0.70, r'$y = ax^{-b} + c$' % r2)
plt.text(450, 0.67, 'R2 score: %.3f' % r2)
plt.text(450, 0.64, r'Reduced $\chi^2$: %.3f' % chisq)
plt.ylabel("AUC")
plt.xlabel("Size")
plt.savefig("results/learning_curve_fit_classification.png", dpi=300)
plt.show()

plt.figure()
plt.errorbar(n, auc_mean, auc_std, None, 'b.', label='data')
n_dense = np.linspace(n[0], n[-1]+100, 100)
plt.plot(n_dense, func(n_dense, *popt), 'r-')
plt.text(450, 0.70, r'$y = ax^{-b} + c$' % r2)
plt.text(450, 0.67, 'R2 score: %.3f' % r2)
plt.text(450, 0.64, r'Reduced $\chi^2$: %.3f' % chisq)
plt.ylabel("AUC")
plt.xlabel("Size")
a = ufloat(popt[0], param_err[0])
b = ufloat(popt[1], param_err[1])
c = ufloat(popt[2], param_err[2])
text_res = "Best fit parameters:\na = {}\nb = {}\nb = {}".format(a, b, c)
print(text_res)
# plotting the model
hires_x = np.linspace(100, 1000, 50)
plt.plot(hires_x, func(hires_x, *popt), 'red')
bound_upper = func(hires_x, *(popt + param_err))
bound_lower = func(hires_x, *(popt - param_err))
# plotting the confidence intervals
plt.fill_between(hires_x, bound_lower, bound_upper,
                 color='red', alpha=0.15)
plt.text(400, 0.4, text_res)
plt.show()
# plt.savefig("results/learning_curve_fit_classification.png", dpi=300)
plt.show()

# test on new data
auc2000 = np.load("results/splitting_method_3/sample_80254/2000_2024-03-25/auc_test.npy")
print(f"sample 2000. Predicted: {func(2000, *popt)} , Actual: {auc2000.mean()}")

auc5000 = np.load("results/splitting_method_3/sample_80254/5000_2024-03-25/auc_test.npy")
print(f"sample 5000. Predicted: {func(5000, *popt)} , Actual: {auc5000.mean()}")

auc10000 = np.load("results/splitting_method_3/sample_80254/10000_2024-03-25/auc_test.npy")
print(f"sample 10.000. Predicted: {func(10000, *popt)} , Actual: {auc10000.mean()}")

plt.figure()
n_dense = np.linspace(n[0], 10100, 100)
plt.plot(n, auc_mean, 'b.', label='fitting data')
plt.plot(2000, auc2000.mean(), 'g.', label="test points")
plt.plot(5000, auc5000.mean(), 'g.', label=None)
plt.plot(10000, auc10000.mean(), 'g.', label=None)
plt.plot(n_dense, func(n_dense, *popt), 'r-', label="fitted curve")
plt.ylabel("AUC")
plt.xlabel("Size")
plt.legend()
plt.savefig("results/learning_curve_test_classification.png", dpi=300)
plt.show()

a, b, c = popt
y_target = 0.98
x_098 = ((y_target - c)/a) ** (-1/b)
print(f"Sample size to get 0.98 AUC: {x_098}")

err = np.zeros(len(popt))
# ci = np.zeros((2, len(popt))
for i in range(len(err)):
    err[i] = np.sqrt(np.sum(r**2)/df * pcov[i, i])


