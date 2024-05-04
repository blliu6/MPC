import numpy as np
import time
import os


class Zones:
    def __init__(self, shape, center=None, r=0.0, low=None, up=None, inner=True):
        self.shape = shape
        self.inner = inner
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2
            self.r = np.sqrt(sum(((self.up - self.low) / 2) ** 2))
        else:
            raise ValueError('error{}'.format(shape))


class Example:
    def __init__(self, n_obs, u_dim, D_zones, I_zones, G_zones, U_zones, f, u, path, dense, units, activation, id, k):
        self.n_obs = n_obs
        self.u_dim = u_dim
        self.D_zones = D_zones
        self.I_zones = I_zones
        self.G_zones = G_zones
        self.U_zones = U_zones
        self.f = f
        self.u = u
        self.path = path
        self.dense = dense
        self.units = units
        self.activation = activation
        self.k = k
        self.id = id
        self.Q = np.eye(n_obs)
        self.R = np.eye(u_dim)
        self.constraint_dim = self.n_obs


class Env:
    def __init__(self, example):
        self.n_obs = example.n_obs
        self.u_dim = example.u_dim
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.G_zones = example.G_zones
        self.U_zones = example.U_zones
        self.f = example.f
        self.path = example.path
        self.u = example.u

        self.dense = example.dense
        self.units = example.units
        self.activation = example.activation
        self.id = example.id
        self.dt = 0.05
        self.k = example.k
        self.Q = np.eye(example.n_obs)
        self.R = np.eye(example.u_dim)
        self.constraint_dim = self.n_obs
        self.dic = dict()


g = 9.8
pi = np.pi
m = 0.1
l = 0.5
mt = 1.1
from numpy import sin, cos, tan


def get_Env(id):
    examples = {
        7: Example(
            n_obs=3,
            u_dim=1,
            D_zones=Zones(shape='box', low=[-5] * 3, up=[5] * 3),
            I_zones=Zones(shape='ball', center=[-0.75, -1, -0.4], r=0.35 ** 2),
            G_zones=Zones(shape='ball', center=[0, 0, 0], r=0.1 ** 2),
            U_zones=Zones(shape='ball', center=[-0.3, -0.36, 0.2], r=0.30 ** 2, inner=True),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u[0],
               ],
            u=3,
            path='ex7/model',
            dense=5,
            units=50,
            activation='relu',
            id=7,
            k=50
        ),  # Academic 3D
        9: Example(
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[-0.51, 0.49], up=[-0.49, 0.51]),
            G_zones=Zones('box', low=[-0.05, -0.05], up=[0.05, 0.05]),
            U_zones=Zones('box', low=[-0.4, 0.2], up=[0.1, 0.35]),
            f=[lambda x, u: x[1],
               lambda x, u: (1 - x[0] ** 2) * x[1] - x[0] + u[0]
               ],
            u=3,
            path='ex9/model',
            dense=5,
            units=64,
            activation='relu',
            id=9,
            k=50
        ),  # Oscillator
        10: Example(
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-6, -7 * pi / 10], up=[6, 7 * pi / 10]),
            I_zones=Zones('box', low=[-1, -pi / 16], up=[1, pi / 16]),
            G_zones=Zones('ball', center=[0, 0], r=0.1 ** 2),
            U_zones=Zones('box', low=[-5, -pi / 2], up=[5, pi / 2], inner=False),
            f=[lambda x, u: sin(x[1]),
               lambda x, u: -u[0]
               ],
            u=3,
            path='ex10/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            k=50
        ),  # Dubins' Car
        12: Example(
            n_obs=3,
            u_dim=1,
            D_zones=Zones('box', low=[-2.2] * 3, up=[2.2] * 3),
            I_zones=Zones('box', low=[-0.2] * 3, up=[0.2] * 3),
            G_zones=Zones('box', low=[-0.1] * 3, up=[0.1] * 3),
            U_zones=Zones('box', low=[-2.2] * 3, up=[-2] * 3),
            f=[lambda x, u: x[1],
               lambda x, u: 30 * sin(x[0]) + 300 * cos(x[0]) * tan(x[2]) + 15 * cos(x[0]) / cos(x[2]) ** 2 * u[0],
               lambda x, u: u[0],
               ],
            u=3,
            path='ex12/model',
            dense=5,
            units=40,
            activation='relu',
            id=12,
            k=50
        ),  # Bicycle Steering
        13: Example(
            n_obs=7,
            u_dim=1,
            D_zones=Zones('box', low=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) - 5,
                          up=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) + 5),
            I_zones=Zones('box', low=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) - 0.05,
                          up=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) + 0.05),
            G_zones=Zones('box', low=np.array([0.87, 0.37, 0.56, 2.75, 0.22, 0.08, 0.27]) - 0.1,
                          up=np.array([0.87, 0.37, 0.56, 2.75, 0.22, 0.08, 0.27]) + 0.1),
            U_zones=Zones('box', low=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) - 4.5,
                          up=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) + 4.5, inner=False),
            f=[lambda x, u: 1.4 * x[2] - 0.9 * x[0],
               lambda x, u: 2.5 * x[4] - 1.5 * x[1] + u[0],
               lambda x, u: 0.6 * x[6] - 0.8 * x[1] * x[2],
               lambda x, u: 2 - 1.3 * x[2] * x[3],
               lambda x, u: 0.7 * x[0] - x[3] * x[4],
               lambda x, u: 0.3 * x[0] - 3.1 * x[5],
               lambda x, u: 1.8 * x[5] - 1.5 * x[1] * x[6],
               ],
            u=0.3,
            path='ex13/model',
            dense=5,
            units=50,
            activation='relu',
            id=13,
            k=50
        ),  # LALO20
        20: Example(  # example 8
            n_obs=4,
            u_dim=1,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
            U_zones=Zones('ball', center=[-2, -2, -2, -2], r=1),
            G_zones=None,
            f=[lambda x, u: -x[0] - x[3] + u[0],
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u[0],
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            u=1,
            path='ex20/model',
            dense=5,
            units=30,
            activation='relu',
            id=20,
            k=100
        )
    }

    return Env(examples[id])
