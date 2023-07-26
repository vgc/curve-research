import bezier as B
import numpy as np
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
        return B.make_cubic_bezier_func(p0, p1, p2, p3)
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
    curvature = (dppx * dpy - dppy * dpx) / (speed * speed * speed)
    return (speed, curvature)

class BezierPlotter():
    def __init__(self):
        self.fig = make_subplots(rows=2, cols=3)

    def plot_bezier(self, bezier, name):
        fig = self.fig
        u = np.linspace(0, 1, 100)
        data = bezier(u)
        speed, curvature = bezier_speed_and_curvature(data)
        accelerations = np.linalg.norm(data[2], axis=1)
        absCurvature = np.abs(curvature)
        fig.add_trace(plgo.Line(x=data[0][:, 0], y=data[0][:, 1], name=f"[{name}] p  "), row=1, col=1)
        fig.add_trace(plgo.Line(x=data[1][:, 0], y=data[1][:, 1], name=f"[{name}] dp "), row=1, col=2)
        fig.add_trace(plgo.Line(x=data[2][:, 0], y=data[2][:, 1], name=f"[{name}] ddp"), row=1, col=3)
        fig.add_trace(plgo.Line(x=u, y=speed, name=f"[{name}] speed"), row=2, col=2)
        fig.add_trace(plgo.Line(x=u, y=absCurvature, name=f"[{name}] curvature"), row=2, col=3)
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

def find_speeds_for_minimum_curvature_max(angle0, angle1, u, speeds, plot=False):
    angle_bound_bezier = make_angle_bound_cubic_bezier(angle0, angle1)
    res = len(speeds)
    grid = np.ndarray(shape=(res, res), dtype=float)
    for i, x in enumerate(speeds):
        for j, y in enumerate(speeds):
            data = angle_bound_bezier(x, y)(u)
            _, curvature = bezier_speed_and_curvature(data)
            curvatureMax = np.max(np.abs(curvature))
            grid[i][j] = curvatureMax
    if plot:
        fig = plgo.Figure(data=[plgo.Surface(z=grid, x=speeds, y=speeds)])
        fig.show()
    coords = np.divmod(grid.argmin(), grid.shape[1])
    minCurvatureMax = grid[coords]
    speed0 = speeds[coords[0]]
    speed1 = speeds[coords[1]]
    return np.array((speed0, speed1, minCurvatureMax))

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
maxMag = 2.5

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
    u = np.linspace(0, 1, 100)
    speeds = np.linspace(minMag, maxMag, 100)

    angle0 = 5.478
    angle1 = -2.98

    sa0, sb0, _ = find_speeds_for_minimum_curvature_max(angle0, angle1, u, speeds, False)
    print(f"tangent lengths: {(sa0, sb0)}")
    sa1, sb1, _ = ogh_curvature_max(angle0, angle1, u)
    print(f"tangent lengths (OGH): {(sa1, sb1)}")

    plotter = BezierPlotter()
    b = make_angle_bound_cubic_bezier(angle0, angle1)
    plotter.plot_bezier(b(sa0, sb0), "minKMax")
    plotter.plot_bezier(b(sa1, sb1), "OGH")
    plotter.show()

if 1:
    u = np.linspace(0, 1, 100)
    speeds = np.linspace(minMag, maxMag, 100)

    res = 40
    a0Space = np.linspace(0, M.pi * 2, res * 2)
    a1Space = np.linspace(0, -M.pi, res)
    #a0Space = np.linspace(1.9, 1.91, res)
    #b0Space = np.linspace(-0.5, -0.51, res)

    try:
        grid0 = np.load(f"grid0_{res}.npy")
    except OSError:
        grid0 = compute_grid(find_speeds_for_minimum_curvature_max, a0Space, a1Space, u, speeds)
        np.save(f"grid0_{res}", grid0, allow_pickle=True, fix_imports=False)

    try:
        grid1 = np.load(f"grid1_{res}.npy")
    except OSError:
        grid1 = compute_grid(ogh2_curvature_max, a0Space, a1Space, u)
        np.save(f"grid1_{res}", grid1, allow_pickle=True, fix_imports=False)

    xaxis=dict(range=[np.min(a0Space), np.max(a0Space)])
    yaxis=dict(range=[np.min(a1Space), np.max(a1Space)])

    fig = plgo.Figure()
    fig.add_surface(z=grid0[:,:,2].T, x=a0Space, y=a1Space, colorscale='YlGnBu', name="minKMax")
    #fig.add_surface(z=grid1[:,:,2].T, x=a0Space, y=a1Space, colorscale='YlOrRd', name="OGH", opacity=0.9)
    #fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    #fig.add_trace(plgo.Surface(z=grid0[:,:,2], x=a0Space, y=a1Space), row=1, col=1)
    #fig.add_trace(plgo.Surface(z=grid1[:,:,2], x=a0Space, y=a1Space), row=1, col=2)
    fig.update_layout(scene = dict(
                    xaxis_title='Angle0',
                    yaxis_title='Angle1',
                    zaxis_title='Curvature Maximum',
                    xaxis=xaxis,
                    yaxis=yaxis,
                    zaxis=dict(range=[np.min(grid0), np.max(grid0)])))
    fig.show()

    fig = plgo.Figure()
    fig.add_surface(z=grid0[:,:,0].T, x=a0Space, y=a1Space, colorscale='YlGnBu', name="Speed0")
    fig.update_layout(scene = dict(
                    xaxis_title='Angle0',
                    yaxis_title='Angle1',
                    zaxis_title='Speed',
                    xaxis=xaxis,
                    yaxis=yaxis,))
    fig.show()
    fig = plgo.Figure()
    fig.add_surface(z=grid0[:,:,1].T, x=a0Space, y=a1Space, colorscale='YlOrRd', name="Speed1")
    fig.update_layout(scene = dict(
                    xaxis_title='Angle0',
                    yaxis_title='Angle1',
                    zaxis_title='Speed',
                    xaxis=xaxis,
                    yaxis=yaxis,))
    fig.show()
