import bezier
import numpy as np
import scipy.optimize as spo
import math as M
import plotly.express as plx
import plotly.graph_objects as plgo
from plotly.subplots import make_subplots

def dirFromAngle(angle):
    return (M.cos(angle), M.sin(angle))

def make_angle_grid_cubic_bezier_maker_from_dirs(v0, v1):
    def angle_bound_cubic_bezier_maker(speed0, speed1):
        return bezier.CubicBezier.fromTangentDirsAndSpeeds(
            (0, 0), (1, 0), v0, v1, speed0, speed1, True)
    return angle_bound_cubic_bezier_maker

def make_cubic_bezier_maker_from_angle_ranges(a0, a1, isGrid=True):
    # todo: grid v0 v1
    v0 = (np.cos(a0), np.sin(a0))
    v1 = (np.cos(a1), np.sin(a1))
    if np.shape(v0[0]):
        v0 = np.column_stack(v0)
        v1 = np.column_stack(v1)
    def angle_bound_cubic_bezier_maker(speed0, speed1):
        return bezier.CubicBezier.fromTangentDirsAndSpeeds(
            (0, 0), (1, 0), v0, v1, speed0, speed1, isGrid)
    return angle_bound_cubic_bezier_maker

def is_bezier_sampling_cusped(data):
    #std::atan2(a.det(b), a.dot(b));
    p, v, a, _ = data
    dpx0 = v[0, :-1]
    dpy0 = v[1, :-1]
    dpx1 = v[0, 1:]
    dpy1 = v[1, 1:]
    y = dpx0*dpy1 - dpy0*dpx1
    x = dpx0*dpx1 + dpy0*dpy1
    angles = np.arctan2(y, x)
    return np.max(np.abs(angles)) > M.pi * 0.2

class BezierPlotter():
    def __init__(self):
        self.fig = make_subplots(rows=2, cols=3)

    def plot_bezier(self, func, u, name):
        u = np.array(u, copy=False)
        fig = self.fig
        p, v, a, _ = data = func(u)
        speed, curvature, _ = bezier.compute_speed_and_curvature(data)
        accelerations = np.linalg.norm(a, axis=1)
        absCurvature = np.abs(curvature)
        fig.add_trace(plgo.Scatter(mode='lines', x=p[0], y=p[1], name=f"[{name}] p  "), row=1, col=1)
        fig.add_trace(plgo.Scatter(mode='lines', x=v[0], y=v[1], name=f"[{name}] dp "), row=1, col=2)
        fig.add_trace(plgo.Scatter(mode='lines', x=a[0], y=a[1], name=f"[{name}] ddp"), row=1, col=3)
        fig.add_trace(plgo.Scatter(mode='lines', x=u, y=speed, name=f"[{name}] speed"), row=2, col=2)
        fig.add_trace(plgo.Scatter(mode='lines', x=u, y=absCurvature, name=f"[{name}] curvature"), row=2, col=3)

        vAngles = np.arctan2(v[1], v[0])
        fig.add_trace(plgo.Scatter(mode='lines', x=u, y=vAngles, name=f"[{name}] vAngles"), row=2, col=1)

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
    _, curvature, _ = bezier.compute_speed_and_curvature(data)
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
        _, curvature, _ = bezier.compute_speed_and_curvature(data)
        curvature = np.abs(curvature)
        if debug:
            print(f"{y}: {curvature}")
        return -curvature

    r = spo.minimize(kfun, np.array((0,)), **opt_args)
    return -kfun(r.x)

def find_speeds_for_minimum_curvature_max(angle0, angle1, u, plot=False, n=30):

    speeds = np.linspace(minMag, maxMag, n)
    dataGrid = make_cubic_bezier_maker_from_angle_ranges(angle0, angle1, True)(speeds, speeds)(u)
    _, kGrid, _ = bezier.compute_speed_and_curvature(dataGrid)
    kGrid = np.abs(kGrid)
    kMaxGrid = np.max(kGrid, axis=-1)
    for idx, _ in np.ndenumerate(kMaxGrid):
        data = (dataGrid[0][idx], dataGrid[1][idx], dataGrid[2][idx], dataGrid[3])
        if is_bezier_sampling_cusped(data):
            kMaxGrid[idx] = KInf

    if plot:
        gridMin = np.min(kMaxGrid)
        gridMax = np.max(kMaxGrid)
        cmax = min(gridMax, gridMin + 5)
        cscale = plx.colors.sequential.Turbo
        cscale.append('#303030')
        fig = plgo.Figure()
        fig = fig.add_contour(
            z=kMaxGrid.T, x=speeds, y=speeds,
            colorscale=cscale,
            zmin=gridMin, zmax=cmax,
            contours_coloring='heatmap',
        )
        fig.update_layout(
            width=1000,
            height=900,
            title=f"Maximum of Curvature of Canonical Cubic Bezier with tangent speeds (x, y) and angles ({angle0}, {angle1})",
            xaxis_title='Speed0',
            yaxis_title='Speed1',
            scene=dict(
                xaxis_title='Speed0',
                yaxis_title='Speed1',
                zaxis_title='Maximum of Curvature ',
                zaxis=dict(
                    range=[0, gridMax]
                )))
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        fig.show()

    coords = np.divmod(kMaxGrid.argmin(), kMaxGrid.shape[1])

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

    angle_bound_bezier = make_cubic_bezier_maker_from_angle_ranges(angle0, angle1, False)
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
    angle_bound_bezier = make_angle_grid_cubic_bezier_maker_from_dirs(v0, v1)
    speed0, speed1 = ogh_speeds(v0, v1)
    data = angle_bound_bezier(speed0, speed1)(u)
    _, curvature, _ = bezier.compute_speed_and_curvature(data)
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
    angle_bound_bezier = make_angle_grid_cubic_bezier_maker_from_dirs(v0, v1)
    speed0, speed1 = ogh_speeds(v0, v1)
    speed0 = clamp_speed(speed0)
    speed1 = clamp_speed(speed1)
    data = angle_bound_bezier(speed0, speed1)(u)
    _, curvature, _ = bezier.compute_speed_and_curvature(data)
    curvatureMax = np.max(np.abs(curvature))
    return np.array((speed0, speed1, curvatureMax))

def compute_grid(method, xSpace, ySpace, *args):
    grid = np.ndarray(shape=(xSpace.shape[0], ySpace.shape[0], 3), dtype=float)
    for i, x in enumerate(xSpace):
        for j, y in enumerate(ySpace):
            print(f"[{i}][{j}]: {method.__name__}({x}, {y}, ...)")
            grid[i][j] = method(x, y, *args)
    return grid

def debugCase(angle0, angle1, plotSpeedKGraph=False, bezierPlotter=None):
    u = np.linspace(0, 1, 100)

    #km: 1.278203867684204, curvatureMax: 1.7430656580501422, x: [0.36114038 0.46252442]
    #sa0, sb0 = 0.36114038, 0.46252442
    sa0, sb0, _ = find_speeds_for_minimum_curvature_max(angle0, angle1, u, plotSpeedKGraph, 200)
    print(f"tangent lengths: {(sa0, sb0)}")
    #sa1, sb1, _ = ogh_curvature_max(angle0, angle1, u)
    #print(f"tangent lengths (OGH): {(sa1, sb1)}")

    if bezierPlotter:
        b = make_cubic_bezier_maker_from_angle_ranges(angle0, angle1, False)
        max_curvature(b, (sa0, sb0), u, True)
        plotter.plot_bezier(b(sa0, sb0), u, "minKMax")
        #plotter.plot_bezier(b(sa1, sb1), u, "OGH")
        #plotter.plot_bezier(b(8, 4.86), u, "test")

if 1:
    plotter = BezierPlotter()
    #debugCase(-0.92, 2.88, True)
    #debugCase(-1.07, 2.88, True)
    debugCase(-2.21, 1.50, True)#, plotter)
    debugCase(-2.56, 1.61, True)
    debugCase(-2.76, 1.99, True)
    #plotter.show()

if 0:
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

    grid0 = grid1

    fig = plgo.Figure()
    grid0Max = np.max(grid0)
    fig.add_contour(
        z=grid0[:,:,2].T, x=a0Space, y=a1Space,
        colorscale='inferno',
        zmin=0, zmax=min(grid0Max, 25),
        #cmin=0, cmax=min(grid0Max, 25),
        name="minKMax")
    #fig.add_surface(z=grid1[:,:,2].T, x=a0Space, y=a1Space, colorscale='YlOrRd', name="OGH", opacity=0.9)
    #fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    #fig.add_trace(plgo.Surface(z=grid0[:,:,2], x=a0Space, y=a1Space), row=1, col=1)
    #fig.add_trace(plgo.Surface(z=grid1[:,:,2], x=a0Space, y=a1Space), row=1, col=2)
    fig.update_layout(
        width=1000,
        height=900,
        scene=dict(
            xaxis_title='Angle0',
            yaxis_title='Angle1',
            zaxis_title='Curvature Maximum',
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=dict(range=[0, min(200, grid0Max * 2)])))
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    #fig.show()

    fig = plgo.Figure()
    fig.add_contour(
        z=grid0[:,:,0].T, x=a0Space, y=a1Space,
        colorscale='turbo',
        zmin=0.2, zmax=3,
        contours=dict(
            start=0.2,
            end=3,
            size=0.05,
        ),
        contours_coloring='heatmap',
        #colorscale='plasma',
        #surfacecolor=grid0[:,:,2].T, cmin=0, cmax=min(grid0Max, 25),
        name="minKMax")
    fig.update_layout(
        width=1000,
        height=900,
        xaxis_title='Angle0',
        yaxis_title='Angle1',
        zaxis_title='Speed0',
        scene=dict(
            xaxis_title='Angle0',
            yaxis_title='Angle1',
            zaxis_title='Speed0',
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=dict(range=[0, 10])
            ))
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    fig.show()

