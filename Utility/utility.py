import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


def linear_utility(x, a, b):
    # This works fine on smaller numbers
    return a + b*x

def quadriatic_utility(x, b):
    return x - (b)*(x**2)

def logarithmic_utility(x, a, b):
    return np.log(a) + b*np.log(x)

def negative_exp_utility(x, c):
    return -np.exp(-(c*x))

def narrow_power(B, x):
    return (B / (B - 1))*(x**(1 - (1/B)))

xdata = np.linspace(0, 10, 100)
negExp = negative_exp_utility(xdata, 0.5)
quad = quadriatic_utility(xdata, 0.5)
narrow_pow = narrow_power(2, xdata)
lin = linear_utility(xdata, 2, 3)

# Create plot
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

ax0.plot(xdata, negExp, label="-exp(-b*Q)")
ax0.set_title("Negative Exponential Utility Curve")
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel("U(Q)")
ax0.legend()
ax0.titlesize = 10

ax1.plot(xdata, quad, label="Q - b*Q^2")
ax1.set_title("Quadratic Utility Curve")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.legend()

ax2.plot(xdata, lin, label="a + b*Q")
ax2.set_title("Linear Utility Curve")
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_ylabel("U(Q)")
ax2.set_xlabel("Q")
ax2.legend()

ax3.plot(xdata, narrow_pow, label="(b / (b-1))*Q^(1-(1/b))")
ax3.set_title("Narrow Power Utility Curve")
ax3.set_yticks([])
ax3.set_xticks([])
ax3.set_xlabel("Q")
ax3.legend()

for _ax in [ax0, ax1, ax2, ax3]:
    _ax.spines["right"].set_visible(False)
    _ax.spines["top"].set_visible(False)

fig.suptitle("Utility measures over increasing quantities of a good")
plt.show()


def cobb_douglas(g1, g2, a1, a2):
    return g1**a1 * g2**a2

fig, ax = plt.subplots()
g1 = np.linspace(1, 10, 100)
g2 = np.linspace(1, 20, 100).reshape((100, 1))
contours = ax.contourf(g1, g2.flatten(), cobb_douglas(g1, g2, .5, .5))
fig.colorbar(contours)
ax.set_xlabel("g1")
ax.set_ylabel("g2")
ax.set_title("Cobb Douglas: g1^(.5)*g2^(1-(.5))")
plt.show()


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
g1, g2 = np.meshgrid(g1, g2)
U = cobb_douglas(g1, g2, .5, .5)

# Plot the surface.
surf = ax.plot_surface(g1, g2, U, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
ax.set_xlabel('g1')
ax.set_ylabel('g2')
ax.set_zlabel('U(g1, g2)')
plt.title("Cobb Douglas Utility Curve for two goods")
plt.show()
