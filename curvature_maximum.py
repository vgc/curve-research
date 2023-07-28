import bezier
import numpy as np
import scipy.optimize as spo
import math as M
import plotly.express as plx
import plotly.graph_objects as plgo
from plotly.subplots import make_subplots

def dirFromAngle(angle):
    return (M.cos(angle), M.sin(angle))

def make_angle_bound_cubic_bezier_from_dirs(v0, v1):
    def angle_bound_cubic_bezier(speed0, speed1):
        p0 = np.array((0, 0))
        p3 = np.array((1, 0))
        p1 = np.array((p0[0] + v0[0] * speed0, p0[1] + v0[1] * speed0))
        p2 = np.array((p3[0] - v1[0] * speed1, p3[1] - v1[1] * speed1))
        return bezier.make_cubic_bezier_func(p0, p1, p2, p3)
    return angle_bound_cubic_bezier

def make_angle_bound_cubic_bezier(a0, a1):
    v0 = dirFromAngle(a0)
    v1 = dirFromAngle(a1)
    return make_angle_bound_cubic_bezier_from_dirs(v0, v1)

def bezier_speed_and_curvature(data):
    dpx = data[1][:, 0]
    dpy = data[1][:, 1]
    dppx = data[2][:, 0]
    dppy = data[2][:, 1]
    speed = np.linalg.norm(data[1], axis=1)
    num = dppx * dpy - dppy * dpx
    den = speed * speed * speed
    curvature = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    return (speed, curvature)

def bezier_tangent_angles(data):
    return np.arctan2(data[1][:, 1], data[1][:, 0])

def is_bezier_sampling_cusped(data):
    #std::atan2(a.det(b), a.dot(b));
    dpx0 = data[1][:-1, 0]
    dpy0 = data[1][:-1, 1]
    dpx1 = data[1][1:, 0]
    dpy1 = data[1][1:, 1]
    y = dpx0*dpy1 - dpy0*dpx1
    x = dpx0*dpx1 + dpy0*dpy1
    angles = np.arctan2(y, x)
    return np.max(np.abs(angles)) > M.pi * 0.2

class BezierPlotter():
    def __init__(self):
        self.fig = make_subplots(rows=2, cols=3)

    def plot_bezier(self, bezier, u, name):
        fig = self.fig
        data = bezier(u)
        speed, curvature = bezier_speed_and_curvature(data)
        accelerations = np.linalg.norm(data[2], axis=1)
        absCurvature = np.abs(curvature)
        fig.add_trace(plgo.Scatter(mode='lines', x=data[0][:, 0], y=data[0][:, 1], name=f"[{name}] p  "), row=1, col=1)
        fig.add_trace(plgo.Scatter(mode='lines', x=data[1][:, 0], y=data[1][:, 1], name=f"[{name}] dp "), row=1, col=2)
        fig.add_trace(plgo.Scatter(mode='lines', x=data[2][:, 0], y=data[2][:, 1], name=f"[{name}] ddp"), row=1, col=3)
        fig.add_trace(plgo.Scatter(mode='lines', x=u, y=speed, name=f"[{name}] speed"), row=2, col=2)
        fig.add_trace(plgo.Scatter(mode='lines', x=u, y=absCurvature, name=f"[{name}] curvature"), row=2, col=3)

        tAngles = bezier_tangent_angles(data)
        fig.add_trace(plgo.Scatter(mode='lines', x=u, y=tAngles, name=f"[{name}] tAngles"), row=2, col=1)

        print(f"sum(|ddp|) for [{name}]: {np.sum(accelerations * accelerations) / u.shape[0]}")
        print(f"ksum for [{name}]: {np.sum(absCurvature * absCurvature) / u.shape[0]}")
        dticks = [0.1, 0.2, 1]
        for i in range(3):
            dtick = dticks[i]
            fig.update_xaxes(
                dtick=dtick,
                row=1, col=1+i
            )
            fig.update_yaxes(
                dtick=dtick,
                scaleanchor=f"x{1+i}",
                scaleratio=1,
                row=1, col=1+i
            )

    def show(self):
        self.fig.show()

KInf = 50000

def max_curvature(angle_bound_bezier, x, u, debug=False):
    b = angle_bound_bezier(x[0], x[1])
    data = b(u)
    _, curvature = bezier_speed_and_curvature(data)
    kmIdx = np.argmax(np.abs(curvature))
    opt_bounds = ((u[max(0, kmIdx - 1)], u[min(len(u) - 1, kmIdx + 1)]),)

    if is_bezier_sampling_cusped(data):
        return KInf

    opt_args = dict(
        method="L-BFGS-B",
        tol=0.0001,
        bounds=opt_bounds,
        options = dict(
            maxiter=200,
            #eps=0.0001,
            #finite_diff_rel_step=0.0001
        )
    )

    def kfun(y):
        data = b(y)
        _, curvature = bezier_speed_and_curvature(data)
        curvature = np.abs(curvature)
        if debug:
            print(f"{y}: {curvature}")
        return -curvature

    r = spo.minimize(kfun, np.array((0,)), **opt_args)
    return -kfun(r.x)

def find_speeds_for_minimum_curvature_max(angle0, angle1, u, plot=False):
    angle_bound_bezier = make_angle_bound_cubic_bezier(angle0, angle1)

    n = 30
    speeds = np.linspace(minMag, maxMag, n)
    grid = np.ndarray(shape=(n, n), dtype=float)
    for i, x in enumerate(speeds):
        for j, y in enumerate(speeds):
            data = angle_bound_bezier(x, y)(u)
            _, curvature = bezier_speed_and_curvature(data)
            curvatureMax = np.max(np.abs(curvature))
            # maximum can be missed by the sampling
            if is_bezier_sampling_cusped(data):
                curvatureMax = KInf
            grid[i][j] = curvatureMax
    if plot:
        gridMin = np.min(grid)
        gridMax = np.max(grid)
        cmax = min(gridMax, gridMin + 50)
        cscale = plx.colors.sequential.Turbo
        cscale.append('#303030')
        fig = plgo.Figure()
        fig = fig.add_contour(
            z=grid, x=speeds, y=speeds, colorscale=cscale, zmin=gridMin, zmax=cmax)
        fig.update_layout(
            scene = dict(
                xaxis_title='Speed0',
                yaxis_title='Speed1',
                zaxis_title='Curvature Maximum',
                zaxis=dict(range=[0, min(gridMax, KInf-1)])))
        fig.show()

    coords = np.divmod(grid.argmin(), grid.shape[1])

    #minCurvatureMax = grid[coords]
    #speed0 = speeds[coords[0]]
    #speed1 = speeds[coords[1]]
    #return np.array((speed0, speed1, minCurvatureMax))

    c0 = np.maximum(0, np.array(coords) - 1)
    c1 = np.minimum(n - 1, np.array(coords) + 1)
    opt_bounds = (
        (speeds[c0[0]], speeds[c1[0]]),
        (speeds[c0[1]], speeds[c1[1]]))

    opt_args = dict(
            method="L-BFGS-B",
            tol=0.01,
            bounds=opt_bounds,
            options = dict(
                maxiter=500,
            )
        )

    def fun(x):
        return max_curvature(angle_bound_bezier, x, u)

    r = spo.minimize(fun, np.array((speeds[coords[0]], speeds[coords[1]])), **opt_args)
    km = fun(r.x)

    return np.array((float(r.x[0]), float(r.x[1]), float(km)))


def ogh_speeds(v0, v1):
    v0v1 = v0[0] * v1[0] + v0[1] * v1[1]
    abDotV0 = v0[0] # ab is (1, 0)
    abDotV1 = v1[0]
    den = 4 - v0v1 * v0v1
    speed0 = (6 * abDotV0 - 3 * abDotV1 * v0v1) / den / 3
    speed1 = (6 * abDotV1 - 3 * abDotV0 * v0v1) / den / 3
    return (speed0, speed1)

def ogh_curvature_max(angle0, angle1, u):
    v0 = dirFromAngle(angle0)
    v1 = dirFromAngle(angle1)
    angle_bound_bezier = make_angle_bound_cubic_bezier_from_dirs(v0, v1)
    speed0, speed1 = ogh_speeds(v0, v1)
    data = angle_bound_bezier(speed0, speed1)(u)
    _, curvature = bezier_speed_and_curvature(data)
    curvatureMax = np.max(np.abs(curvature))
    return np.array((speed0, speed1, curvatureMax))

def clamp(x, a, b):
   return max(min(x, b), a)

minMag = 0.2
maxMag = 3

def clamp_speed(x):
   return clamp(x, minMag, maxMag)

def ogh2_curvature_max(angle0, angle1, u):
    v0 = dirFromAngle(angle0)
    v1 = dirFromAngle(angle1)
    angle_bound_bezier = make_angle_bound_cubic_bezier_from_dirs(v0, v1)
    speed0, speed1 = ogh_speeds(v0, v1)
    speed0 = clamp_speed(speed0)
    speed1 = clamp_speed(speed1)
    data = angle_bound_bezier(speed0, speed1)(u)
    _, curvature = bezier_speed_and_curvature(data)
    curvatureMax = np.max(np.abs(curvature))
    return np.array((speed0, speed1, curvatureMax))

def compute_grid(method, xSpace, ySpace, *args):
    grid = np.ndarray(shape=(xSpace.shape[0], ySpace.shape[0], 3), dtype=float)
    for i, x in enumerate(xSpace):
        for j, y in enumerate(ySpace):
            print(f"[{i}][{j}]: {method.__name__}({x}, {y}, ...)")
            grid[i][j] = method(x, y, *args)
    return grid

if 0:
    u = np.linspace(0, 1, 1000)

    #angle0 = 4.704496
    #angle1 = -0.4759989

    angle0 = 0.0
    angle1 = -0.5711986642890533

    #km: 1.278203867684204, curvatureMax: 1.7430656580501422, x: [0.36114038 0.46252442]
    sa0, sb0 = 0.36114038, 0.46252442
    #sa0, sb0, _ = find_speeds_for_minimum_curvature_max(angle0, angle1, u, True)
    print(f"tangent lengths: {(sa0, sb0)}")
    sa1, sb1, _ = ogh_curvature_max(angle0, angle1, u)
    print(f"tangent lengths (OGH): {(sa1, sb1)}")

    plotter = BezierPlotter()
    b = make_angle_bound_cubic_bezier(angle0, angle1)

    max_curvature(b, (sa0, sb0), True)
    plotter.plot_bezier(b(sa0, sb0), u, "minKMax")
    #plotter.plot_bezier(b(sa1, sb1), u, "OGH")
    #plotter.plot_bezier(b(8, 4.86), u, "test")

    plotter.show()

if 1:
    u = np.linspace(0, 1, 100)

    hn = 200
    n = hn * 2 - 1
    a0Space = np.linspace(0, M.pi * 2, n)
    a1Space = np.linspace(0, M.pi, hn)
    #a0Space = np.linspace(1.9, 1.91, res)
    #b0Space = np.linspace(-0.5, -0.51, res)

    useCache = True

    try:
        if not useCache:
            raise OSError
        grid0 = np.load(f"grid0_{n}.npy")
    except OSError:
        grid0 = compute_grid(find_speeds_for_minimum_curvature_max, a0Space, a1Space, u)
        if useCache:
            np.save(f"grid0_{n}", grid0, allow_pickle=True, fix_imports=False)

    try:
        if not useCache:
            raise OSError
        grid1 = np.load(f"grid1_{n}.npy")
    except OSError:
        grid1 = compute_grid(ogh2_curvature_max, a0Space, a1Space, u)
        if useCache:
            np.save(f"grid1_{n}", grid1, allow_pickle=True, fix_imports=False)

    # complete grids by symmetry
    a1Space = np.linspace(0, M.pi * 2, n)
    grid0 = np.hstack((grid0, grid0[::-1,-2::-1]))
    grid1 = np.hstack((grid1, grid1[::-1,-2::-1]))

    # extend by repetition to [-pi, 2pi]
    a0Space = np.linspace(-M.pi, M.pi * 2, n + hn - 1)
    a1Space = np.linspace(-M.pi, M.pi * 2, n + hn - 1)
    grid0 = np.hstack((grid0[:,hn:-1], grid0))
    grid1 = np.hstack((grid1[:,hn:-1], grid1))
    grid0 = np.vstack((grid0[hn:-1,:], grid0))
    grid1 = np.vstack((grid1[hn:-1,:], grid1))

    tickvals = [-M.pi, -M.pi * 0.5, 0, M.pi * 0.5, M.pi, M.pi * 1.5, M.pi * 2],
    ticktext = ['$\pi$', '$-\frac{1}{2}\pi$', '0', '$\frac{1}{2}\pi$', '\pi', '$\frac{3}{2}\pi$', '2\pi']
    xaxis=dict(
        range=[np.min(a0Space), np.max(a0Space)],
        tickvals=tickvals,
        ticktext=ticktext
    )
    yaxis=dict(
        range=[np.min(a1Space), np.max(a1Space)],
        tickvals=tickvals,
        ticktext=ticktext
    )

    fig = plgo.Figure()
    grid0Max = np.max(grid0)
    fig.add_contour(z=grid0[:,:,2].T, x=a0Space, y=a1Space,
                    colorscale='inferno',
                    zmin=0, zmax=min(grid0Max, 25),
                    #cmin=0, cmax=min(grid0Max, 25),
                    name="minKMax")
    #fig.add_surface(z=grid1[:,:,2].T, x=a0Space, y=a1Space, colorscale='YlOrRd', name="OGH", opacity=0.9)
    #fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    #fig.add_trace(plgo.Surface(z=grid0[:,:,2], x=a0Space, y=a1Space), row=1, col=1)
    #fig.add_trace(plgo.Surface(z=grid1[:,:,2], x=a0Space, y=a1Space), row=1, col=2)
    fig.update_layout(
        scene = dict(
            xaxis_title='Angle0',
            yaxis_title='Angle1',
            zaxis_title='Curvature Maximum',
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=dict(range=[0, min(200, grid0Max * 2)])))
    fig.show()

    fig = plgo.Figure()
    fig.add_contour(z=grid0[:,:,0].T, x=a0Space, y=a1Space,
                    colorscale='rdbu',
                    name="minKMax")
    fig.update_layout(
        scene = dict(
            xaxis_title='Angle0',
            yaxis_title='Angle1',
            #zaxis_title='Speed0',
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=dict(range=[0, 10])
            ))
    fig.show()

