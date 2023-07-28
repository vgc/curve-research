import numpy as np

class CubicBezier():
    # p0, p1, p2, and p3 must broadcastable to each other
    def __init__(self, p0, p1, p2, p3, gridTangents=False):
        self.p0 = np.array(p0, copy=False)
        self.p1 = p1 = np.array(p1, copy=False)
        self.p2 = p2 = np.array(p2, copy=False)
        self.p3 = np.array(p3, copy=False)
        if gridTangents:
            self.p1 = np.repeat(p1[:, np.newaxis, ...], p2.shape[0], 1)
            self.p2 = np.repeat(p2[np.newaxis, ...], p1.shape[0], 0)

    # pA, pB, vAngleA, and vAngleB must be broadcastable against each other.
    # speedA, and speedB must have the same shape.
    @staticmethod
    def fromTangentDirsAndSpeeds(
        pA, pB, vDirA, vDirB, speedA, speedB, gridTangents=False):

        p1 = pA + np.multiply.outer(speedA, vDirA)
        p2 = pB - np.multiply.outer(speedB, vDirB)
        return CubicBezier(pA, p1, p2, pB, gridTangents)

    # pA, pB, vAngleA, and vAngleB must be broadcastable against each other.
    # speedA, and speedB must have the same shape.
    @staticmethod
    def fromTangentAnglesAndSpeeds(
        pA, pB, vAngleA, vAngleB, speedA, speedB, gridTangents=False):

        vDirA = (np.cos(vAngleA), np.sin(vAngleA))
        vDirB = (np.cos(vAngleB), np.sin(vAngleB))
        if np.shape(vDirA[0]):
            vDirA = np.column_stack(vDirA)
            vDirB = np.column_stack(vDirB)

        p1 = pA + np.multiply.outer(speedA, vDirA)
        p2 = pB - np.multiply.outer(speedB, vDirB)
        return CubicBezier(pA, p1, p2, pB, gridTangents)

    def __call__(self, t):
        p0, p1, p2, p3 = self.p0, self.p1, self.p2, self.p3

        # if t is an array, let's make it a np.array
        if np.shape(t):
            t = np.array(t, copy=False)

        u = 1 - t
        t2 = t * t
        t3 = t2 * t
        u2 = u * u
        u3 = u2 * u

        d1 = (p1 - p0, p2 - p1, p3 - p2)
        d2 = (d1[1] - d1[0], d1[2] - d1[1])

        p = ( np.multiply.outer(p0, u3)
            + np.multiply.outer(p1, 3 * u2 * t)
            + np.multiply.outer(p2, 3 * u * t2)
            + np.multiply.outer(p3, t3))

        v = ( np.multiply.outer(d1[0], 3 * u2)
            + np.multiply.outer(d1[1], 6 * u * t)
            + np.multiply.outer(d1[2], 3 * t2))

        a = ( np.multiply.outer(d2[0], 6 * u)
            + np.multiply.outer(d2[1], 6 * t))

        return (p, v, a, t)

def compute_speed_and_curvature(data):
    _, v, a, t = data
    coordAxis= -2 if np.shape(t) else -1
    vx = v.take(indices=0, axis=coordAxis)
    vy = v.take(indices=1, axis=coordAxis)
    ax = a.take(indices=0, axis=coordAxis)
    ay = a.take(indices=1, axis=coordAxis)
    speed = np.linalg.norm(data[1], axis=coordAxis)
    num = ax * vy - ay * vx
    den = speed * speed * speed
    curvature = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    return (speed, curvature, t)

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
