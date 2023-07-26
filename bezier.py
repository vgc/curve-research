import numpy as np

def cubic_bezier(t, p0, p1, p2, p3):
    u = 1 - t
    u = np.array(u)[..., np.newaxis]
    t = np.array(t)[..., np.newaxis]
    t2 = t * t
    t3 = t2 * t
    u2 = u * u
    u3 = u2 * u
    p = u3 * p0 + 3 * u2 * t * p1 + 3 * u * t2 * p2 + t3 * p3
    dp = 3 * u2 * (p1 - p0) + (6 * u * t) * (p2 - p1) + (3 * t2) * (p3 - p2)
    ddp = 6 * u * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
    return (p, dp, ddp)

def make_cubic_bezier_func(p0, p1, p2, p3):
    def bound_cubic_bezier(t):
        return cubic_bezier(t, p0, p1, p2, p3)
    return bound_cubic_bezier

def quad_bezier(t, p0, p1, p2):
    u = 1 - t
    u = np.array(u)[..., np.newaxis]
    t = np.array(t)[..., np.newaxis]
    t2 = t * t
    u = 1 - t
    u2 = u * u
    p = p1 + u2 * (p0 - p1) + t2 * (p2 - p1)
    dp = 2 * u * (p1 - p0) + 2 * t * (p2 - p1)
    #print(f"u:{dp}")
    #print(f"dp:{dp}")
    ddp = 2 * (p2 - 2 * p1 + p0)
    return (p, dp, ddp)

def make_quad_bezier_func(p0, p1, p2):
    def bound_quad_bezier(t):
        return quad_bezier(t, p0, p1, p2)
    return bound_quad_bezier

# [(1 - t)(x1 - x0) + t(x2 - x1)]²
# (1 - t)²(x1 - x0)² + 2(1 - t)(x1 - x0)t(x2 - x1) + t²(x2 - x1)²
# (x1 - x0)² - 2t(x1 - x0)² + t²(x1 - x0)² + 2t(x1 - x0)(x2 - x1) - 2t²(x1 - x0)(x2 - x1) + t²(x2 - x1)²
# [(x1 - x0)² - 2(x1 - x0)(x2 - x1) + (x2 - x1)²]t² + [2(x1 - x0)(x2 - x1) - 2(x1 - x0)²]t + [(x1 - x0)²]
# a = x1 - x0
# b = x2 - x1
# (a - b)²t² + (2ab - 2a²)t + a²
# (c - d)²t² + (2cd - 2c²)t + c²

# A = (a - b)² + (c - d)²
# B = (2ab - 2a²) + (2cd - 2c²)
# C = a² + c²

# ((a - b)² + (c - d)²)*x² + ((2ab - 2a²) + (2cd - 2c²))*x + (a² + c²)

# integral2 sqrt(A x^2 + B x + C) dx = ((2 A x + B) sqrt(x (A x + B) + C))/(2 A) - ((B^2 - 4 A C) log(2 sqrt(A) sqrt(x (A x + B) + C) + 2 A x + B))/(4 A^(3/2)) + constant

def quad_bezier_analytic_arclen(p0, p1, p2):
    a = p1[0] - p0[0]
    b = p2[0] - p1[0]
    c = p1[1] - p0[1]
    d = p2[1] - p1[1]
    amb = a - b
    cmd = c - d
    a2 = a * a
    c2 = c * c
    A = amb * amb + cmd * cmd
    B = -2 * (a2 - a * b + c2 - c * d)
    C = a2 + c2
    ABC1_2 = np.sqrt(A + B + C)
    A1_2 = np.sqrt(A)
    C1_2 = np.sqrt(C)
    A3_2 = A * A1_2
    D = A + A + B
    R1 = ((D * ABC1_2) - B * C1_2) / (A + A)
    R2 = (B * B - 4 * A * C) / (4 * A3_2)
    R3 = (2 * A1_2 * ABC1_2 + D) / (2 * A1_2 * C1_2 + B)
    #print(f"R1, R2, R3: {(R1, R2, R3)}")
    return R1 - R2 * np.log(R3)
