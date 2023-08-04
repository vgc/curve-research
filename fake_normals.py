import bezier
import numpy as np
import scipy.optimize as spo
import math as M
import plotly.express as plx
import plotly.graph_objects as plgo
from plotly.subplots import make_subplots

pA = (0, 0)
pA0 = (7, 1)
pA1 = (12, 4)
pB = (14, 6)
pB0 = (15, 7)
pB1 = (14, 8)
pC = (12, 10)
t0 = (1, 0)
t1 = (1, 0)
t2 = (0, 1)
t3 = (0, 1)

b0 = bezier.CubicBezier(pA, pA0, pA1, pB)
b1 = bezier.CubicBezier(pB, pB0, pB1, pC)

def proto(u, dt0, dt1):
    bt = bezier.CubicBezier(0, dt0 / 3, 1 - dt1 / 3, 1)
    t, dt, _, _ = bt(u)
    print(f"dt[0]:{dt[0]}, dt[-1]:{dt[-1]}")
    return t

def proto2(u, dt0, dt1):
    bt0 = bezier.CubicBezier(0, dt0 / 3, 2 / 3, 1)
    bt1 = bezier.CubicBezier(0, 1 / 3, 1 - dt1 / 3, 1)
    result = np.where(u < 0.5, bt0(u * 2)[0] * 0.5, bt1((u - 0.5) * 2)[0] * 0.5 + 0.5)
    print(result)
    return result

# plotting

def plotStrokes(beziers, r0, r1, u):
    numSegs = len(beziers)
    fig = plgo.Figure()
    sr = np.linspace(r0, r1, numSegs + 1)
    dataArray = tuple(b(u) for b in beziers)
    scArray = tuple(bezier.compute_speed_and_curvature(data) for data in dataArray)
    dnlArray = tuple(s * k for s, k, _ in scArray)

    totalS = 0
    for data in dataArray:
        d = np.diff(data[0], axis=-1)
        dl = np.linalg.norm(d, axis=0)
        curveS = np.sum(dl)
        totalS += curveS
    print("totalS:", totalS)

    dw_ds = (r1 - r0) / totalS

    for i, b in enumerate(beziers):
        p, v, a, _ = dataArray[i]
        speed, curvature, _ = scArray[i]
        sr0, sr3 = sr[i], sr[i + 1]

        print(f"cp: {b.cps()}")
        print(f"v: ({v[:,0]}, {v[:,-1]})")
        print(f"a: ({a[:,0]}, {a[:,-1]})")
        print(f"s: ({speed[0]}, {speed[-1]})")
        print(f"k: ({curvature[0]}, {curvature[-1]})")

        btr = bezier.CubicBezier(sr0, sr0 + dw_ds * (speed[0] / 3), sr3 - dw_ds * (speed[-1] / 3), sr3)
        r = btr(u)[0]
        #print(f"r: {r}")

        #r = np.interp(u, (0, 1), (sr0, sr3))

        n = np.array((-v[1] / speed, v[0] / speed))

        o = p + n * (r + 0.002)
        ob = p - n * (r + 0.002)

        rc = np.linspace(0, 1, 11)

        dt0, dt1 = 1, 1
        if i > 0:
            dnlOther = dnlArray[i - 1][1]
            dnl = dnlArray[i][0]
            #if dnlOther != 0 :
            #    dt0 = (dnl + dnlOther) / (2 * dnlOther)
            if dnl * dnlOther < 0:
                # sign of curvature is different, it is impossible to make k1 match k2 with a positive coefficient
                # thus we force both sides to have a fake curvature of 0 (fake C1 inflexion point).
                dt0 = 0
            elif abs(dnlOther) < abs(dnl):
                dt0 = dnlOther / dnl
            #else:
            #    dt0 = dnl / dnlOther
        if i < numSegs - 1:
            #print("s", scArray[i][0][-1], scArray[i + 1][0][0])
            #print("k", scArray[i][1][-1], scArray[i + 1][1][0])
            dnlOther = dnlArray[i + 1][0]
            dnl = dnlArray[i][1]
            #if dnlOther != 0:
            #    dt1 = (dnl + dnlOther) / (2 * dnlOther)
            if dnl * dnlOther < 0:
                dt1 = 0
            elif abs(dnlOther) < abs(dnl):
                dt1 = dnlOther / dnl
            #else:
            #    dt1 = dnl / dnlOther

        t = proto(u, dt0, dt1)

        _, v2, _, _ = data2 = b(t)
        speed2, _, _ = bezier.compute_speed_and_curvature(data2)
        n2 = np.array((-v2[1] / speed2, v2[0] / speed2))
        o2 = p + n2 * r
        o2b = p + n2 * -r
        o2s = p[np.newaxis, ...] + n2[np.newaxis, ...] * np.multiply.outer(rc, r[np.newaxis, :])

        if i == 0:
            mergedOs = o2s
        else:
            print(f"o2s.shape:{o2s.shape}")
            mergedOs = np.concatenate((mergedOs, o2s[...,1:]), axis=-1)

        #for j in range(0, len(p[0]), 8):
        #    fig.add_scatter(
        #        x=(p[0][j], o[0][j]),
        #        y=(p[1][j], o[1][j]),
        #        mode='lines',
        #        line=dict(color='rgb(20, 20, 20)', width=1),
        #        showlegend=False)
        numPoints = len(p[0])
        if i == 1:
            numPoints = int(numPoints / 4)
        for j in range(0, numPoints, 8):
            fig.add_scatter(
                x=(o2[0][j], o2b[0][j]),
                y=(o2[1][j], o2b[1][j]),
                line={
                    'color': 'rgb(30, 30, 150)',
                    'width': (2 if j in (0, numPoints - 1) else 1)},
                mode='lines',
                showlegend=False)

        fig.add_scatter(mode='lines', line={'width': 4}, x=p[0], y=p[1], name=f"centerline {i}")
        #fig.add_scatter(mode='lines',
        #                x=list(o[0]) + [None] + list(ob[0]),
        #                y=list(o[1]) + [None] + list(ob[1]),
        #                name=f"actual geometric offsetline {i}")
        fig.add_scatter(mode='lines',
                        x=list(o2[0]) + [None] + list(o2b[0]),
                        y=list(o2[1]) + [None] + list(o2b[1]),
                        name=f"fake normals offsetline {i}")

    for i, o in enumerate(mergedOs):
        fig.add_scatter(mode='lines',
                            x=o[0],
                            y=o[1],
                            name=f"fake normals offsetline (w*={rc[i]})")

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()

u = np.linspace(0, 1, 600)
plotStrokes((b0, b1), 0.2, 6.0, u)
