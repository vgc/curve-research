import bezier
import numpy as np
import scipy.optimize as spo
import math as M
import plotly.express as plx
import plotly.graph_objects as plgo
from plotly.subplots import make_subplots

pA = (0, 0)
pB = (2, 0)
pC = (3, 1)
pD = (4, 3)
t0 = (1, 0)
t1 = (1, 0)
t2 = (0, 1)
t3 = (0, 1)

b0 = bezier.CubicBezier.fromTangentDirsAndSpeeds(pA, pB, t0, t1, 0.66, 0.66)
b1 = bezier.CubicBezier.fromTangentDirsAndSpeeds(pB, pC, t1, t2, 0.66, 0.66)
b2 = bezier.CubicBezier.fromTangentDirsAndSpeeds(pC, pD, t2, t3, 0.66, 0.66)

def proto(u, dt0, dt1):
    bt = bezier.CubicBezier(0, dt0 / 3, 1 - dt1 / 3, 1)
    return bt(u)[0]

def proto2(u, dt0, dt1):
    bt0 = bezier.CubicBezier(0, dt0 / 3, 2 / 3, 1)
    bt1 = bezier.CubicBezier(0, 1 / 3, 1 - dt1 / 3, 1)
    result = np.where(u < 0.5, bt0(u * 2)[0] * 0.5, bt1((u - 0.5) * 2)[0] * 0.5 + 0.5)
    print(result)
    return result

# plotting

def plotStrokes(beziers, r0, r1, u):
    n = len(beziers)
    fig = plgo.Figure()
    sr = np.linspace(r0, r1, n + 1)
    dataArray = tuple(b(u) for b in beziers)
    scArray = tuple(bezier.compute_speed_and_curvature(data) for data in dataArray)
    dnlArray = tuple(abs(s * k) for s, k, _ in scArray)
    for i, b in enumerate(beziers):
        p, v, a, _ = dataArray[i]
        speed, curvature, _ = scArray[i]
        r = np.interp(u, (0, 1), (sr[i], sr[i + 1]))
        o = p + np.array((-v[1] / speed, v[0] / speed)) * (np.array(r) + 0.002)
        ob = p - np.array((-v[1] / speed, v[0] / speed)) * (np.array(r) + 0.002)
        dt0, dt1 = 1, 1
        if i > 0:
            dnlOther = dnlArray[i - 1][1]
            dnl = dnlArray[i][0]
            #if dnlOther != 0 :
            #    dt0 = (dnl + dnlOther) / (2 * dnlOther)
            if dnlOther < dnl:
                dt0 = dnlOther / dnl
            else:
                dt0 = dnl / dnlOther
        if i < n - 1:
            #print("s", scArray[i][0][-1], scArray[i + 1][0][0])
            #print("k", scArray[i][1][-1], scArray[i + 1][1][0])
            dnlOther = dnlArray[i + 1][0]
            dnl = dnlArray[i][1]
            #if dnlOther != 0:
            #    dt1 = (dnl + dnlOther) / (2 * dnlOther)
            if dnlOther < dnl:
                dt1 = dnlOther / dnl
            else:
                dt1 = dnl / dnlOther
        t = proto(u, dt0, dt1)
        _, v2, _, _ = data2 = b(t)
        speed2, _, _ = bezier.compute_speed_and_curvature(data2)
        o2 = p + np.array((-v2[1] / speed2, v2[0] / speed2)) * np.array(r)
        o2b = p - np.array((-v2[1] / speed2, v2[0] / speed2)) * np.array(r)
        #for j in range(0, len(p[0]), 8):
        #    fig.add_scatter(
        #        x=(p[0][j], o[0][j]),
        #        y=(p[1][j], o[1][j]),
        #        mode='lines',
        #        line=dict(color='rgb(20, 20, 20)', width=1),
        #        showlegend=False)
        for j in range(0, len(p[0]), 8):
            fig.add_scatter(
                x=(o2[0][j], o2b[0][j]),
                y=(o2[1][j], o2b[1][j]),
                mode='lines',
                line=dict(color='rgb(30, 30, 150)', width=1),
                showlegend=False)
        fig.add_scatter(mode='lines', line={'width': 4}, x=p[0], y=p[1], name=f"centerline {i}")
        fig.add_scatter(mode='lines',
                        x=list(o[0]) + [None] + list(ob[0]),
                        y=list(o[1]) + [None] + list(ob[1]),
                        name=f"actual geometric offsetline {i}")
        fig.add_scatter(mode='lines',
                        x=list(o2[0]) + [None] + list(o2b[0]),
                        y=list(o2[1]) + [None] + list(o2b[1]),
                        name=f"fake normals offsetline {i}")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()

u = np.linspace(0, 1, 200)
plotStrokes((b0, b1, b2), 0.2, 2.0, u)
