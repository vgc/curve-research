import numpy as np
import scipy as sp
import plotly.express as plx
import bezier

# Testing arclength computation algorithms.
#
# References:
# https://raphlinus.github.io/curves/2018/12/28/bezier-arclength.html

def wsum(f, r):
    return np.sum(np.multiply(f(r[0]), r[1]))

def make_lg_integrator(n):
    r = np.polynomial.legendre.leggauss(n)
    def wsum_lgn(f, a, b):
        c = 0.5 * (a + b)
        l = (b - a)
        d = 0.5 * l
        def f2(x):
            t = c + d * x
            #print(f"x: {x}")
            #print(f"t: {t}")
            return f(t)
        return wsum(f2, r) * l / 2
    return wsum_lgn

def make_quad_bezier_ds_func(p0, p1, p2):
    def bound_quad_bezier_ds(t):
        _, dp, _ = bezier.quad_bezier(t, p0, p1, p2)
        return np.sqrt(np.sum(dp * dp, -1))
    return bound_quad_bezier_ds

def plot_quad_integrator_error(integrator, resolution):
    grid = np.ndarray(shape=(resolution, resolution), dtype=float)
    p0 = np.array((-1, 0))
    p2 = np.array(( 1, 0))
    SPACE = 5
    invrel = 1 / resolution
    for x in range(resolution):
        for y in range(resolution):
            p1 = np.array((x * invrel * SPACE, y * invrel * SPACE))
            f = make_quad_bezier_ds_func(p0, p1, p2)
            #v0a, e0a = sp.integrate.quad(f, 0, 1, epsabs=1e-15, epsrel=1e-15, limit=10000)
            #v0b, e0b = sp.integrate.quadrature(f, 0, 1, tol=1e-15, maxiter=100, vec_func=False)
            v1 = integrator(f, 0, 1)
            v2 = bezier.quad_bezier_analytic_arclen(p0, p1, p2)
            #print(f"x:{x}")
            #err = np.abs(v1 - v0a)
            #print(f"v0a, v2: {(v0a, v2)}")
            errB = np.abs(v2 - v1)
            #print(f"v0, v1, e1: {v0}, {v1}, {e1}")
            grid[resolution - 1 - y, x] = np.log10(errB) if errB > 0 else -17
            #grid[x, y] = 10 if (v2 > 10) else v2
    fig = plx.imshow(grid)
    fig.show()

plot_quad_integrator_error(make_lg_integrator(24), 2048)
