from __future__ import annotations
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class ParametricCurve2d:
    def __init__(self, x, y, param=sp.Symbol("t")):
        self.x = x
        self.y = y
        self.param = param

        self._x_lambda = sp.lambdify(param, x)
        self._y_lambda = sp.lambdify(param, y)

        # Define self._derive lazily to avoid inifinite recursion.
        self._deriv = None

    def __repr__(self) -> str:
        return f"ParametricCurve({self.x}, {self.y})"

    def __call__(self, t: float):
        return np.array([self._x_lambda(t), self._y_lambda(t)], dtype=float)

    def norm(self, t: float) -> float:
        return np.linalg.norm(self(t))

    @property
    def deriv(self) -> ParametricCurve2d:
        # Define self._deriv on-demand.
        if self._deriv is None:
            self._deriv = ParametricCurve2d(
                sp.diff(self.x), sp.diff(self.y), param=self.param
            )
        return self._deriv


class Cycloid:
    def __init__(
        self,
        base_curve: ParametricCurve2d,
        radius: float = 1,
        begin: float = 0,
        end: float = 10,
        num: int = 50,
    ):
        self.base_curve = base_curve
        self.radius = radius
        self.begin = begin
        self.end = end
        self.num = num

        self._points = np.empty((num, 3), dtype=float)

        t_prev = begin
        theta = 0
        for i, t in enumerate(np.linspace(begin, end, num)):
            tangent = base_curve.deriv(t) / base_curve.deriv.norm(t)
            rotate = np.array(
                [
                    [-np.sin(theta), np.cos(theta) - 1],
                    [1 - np.cos(theta), -np.sin(theta)],
                ]
            )
            self._points[i] = t, *(base_curve(t) + radius * rotate @ tangent)
            theta += integrate.quad(base_curve.deriv.norm, t_prev, t)[0] / radius
            t_prev = t

    def __getitem__(self, key):
        return self._points[key]

    def __repr__(self):
        return f"Cycloid({self.base_curve}, )"

    def save(self, filename: str, delimeter: str = "\t") -> None:
        fmt = "{t}{delimeter}{x}{delimeter}{y}\n"
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write(fmt.format(t="t", x="x", y="y", delimeter=delimeter))
            for t, x, y in self._points:
                fout.write(fmt.format(t=t, x=x, y=y, delimeter=delimeter))


def plot_curves(
    curve_samples, colors, filename: str, save: bool = True, show: bool = False
) -> None:
    """Show `curve_samples` via matplotlib."""
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for i, curve in enumerate(curve_samples):
        ax.plot(curve[:, 1], curve[:, 2], color=colors[i], linewidth=0.5)
    ax.set_aspect(1)

    if filename.endswith(".png"):
        plt.savefig(filename, dpi=500, bbox_inches="tight")
    else:
        plt.savefig(filename, bbox_inches="tight")

    if show:
        plt.show()
    plt.clf()


# Constants
RADIUS = 1
START = 0
END = 6.97
NUM = 200

_t = sp.Symbol("t")
circle = ParametricCurve2d(sp.sin(_t), sp.cos(_t))
epicycloid = Cycloid(circle, radius=RADIUS, begin=START, end=END, num=NUM)

circle_samp = np.column_stack(
    (epicycloid[:, 0], np.vstack([circle(t) for t in epicycloid[:, 0]]))
)

# Save data for other visualization tools, e.g., TikZ.
epicycloid.save("cycloid.dat")

# Save fig
plot_curves([circle_samp, epicycloid], ["r", "k"], "cycloid.png")
