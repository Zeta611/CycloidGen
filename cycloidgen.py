# -*- coding: utf-8 -*-

from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.Symbol('x')
sp.init_printing(use_unicode=True)

class ParametricCurve:
    def __init__(self, u, v):
        self.u_sym = u
        self.v_sym = v
        self.u = sp.lambdify(x, u)
        self.v = sp.lambdify(x, v)
        self.equation = lambda t: np.array([self.u(t), self.v(t)])
        self.norm = lambda t: np.linalg.norm(self.equation(t))

    def __repr__(self):
        return "ParametricCurve(%s, %s)" % (self.u_sym, self.v_sym)

    def derivative(self):
        u_derivative = sp.diff(self.u_sym)
        v_derivative = sp.diff(self.v_sym)
        return ParametricCurve(u_derivative, v_derivative)


_cumulated_sum = 0
_t_previous = None
def cycloid(curve, radius=1, start=0):
    assert isinstance(curve, ParametricCurve)
    def theta(t):
        global _cumulated_sum, _t_previous
        if _t_previous is None:
            _cumulated_sum \
                = integrate.quad(curve.derivative().norm, start, t)[0] / radius
            _t_previous = t
            return _cumulated_sum
        _cumulated_sum \
            += integrate.quad(curve.derivative().norm, _t_previous, t)[0] \
                / radius
        _t_previous = t
        return _cumulated_sum

    def position(t):
        tangent = curve.derivative().equation(t) / curve.derivative().norm(t)
        angle = theta(t)
        transform = np.array([[-np.sin(angle), np.cos(angle) - 1],
                              [1 - np.cos(angle), -np.sin(angle)]])
        return curve.equation(t) + radius * np.dot(transform, tangent)
    return position


a = 1
k = 3
radius = 1 / 2
curve = ParametricCurve(sp.sin(x), 2 * sp.cos(x))
#  curve = ParametricCurve(x, a * sp.sin(k * x))

#  cycloid(curve, radius)(0.9)

start = 0
cycles = 2.5
samples = int(100 * cycles)
cycloid_samples_x = np.zeros(samples)
cycloid_samples_y = np.zeros(samples)
curve_samples_x = np.zeros(samples)
curve_samples_y = np.zeros(samples)

# Save data for other visualization tools, e.g., TikZ.
delimeter = ' '
with open("cycloid.dat", "w") as fout:
    fout.write("% cycloid_x cycloid_y t\n")
    i = 0
    #  for t in np.linspace(start, start + 2 * np.pi * radius * cycles, num=samples):
    for t in np.linspace(start, start + 6.97, num=samples):
        cycloid_coord = cycloid(curve, radius=radius, start=start)(t)
        curve_coord = curve.equation(t)
        fout.write(str(cycloid_coord[0]) + delimeter + str(cycloid_coord[1]) + delimeter + str(t) + '\n')
        cycloid_samples_x[i], cycloid_samples_y[i] = cycloid_coord
        curve_samples_x[i], curve_samples_y[i] = curve_coord
        i += 1

# Show `curve` via matplotlib.
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(cycloid_samples_x, cycloid_samples_y, color='r', linewidth=0.5)
ax.plot(curve_samples_x, curve_samples_y, color='k', linewidth=0.5)
ax.set_aspect(1)
#  plt.savefig("cycloid.pdf", bbox_inches="tight")
plt.savefig("cycloid.png", dpi=500, bbox_inches="tight")
#  plt.show()
plt.clf()
