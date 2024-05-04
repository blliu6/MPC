import casadi as ca
import numpy as np
from Env import get_Env
import math


class MPC:
    N = 10  # MPC步长大小
    eps = 2e-2  # 误差控制

    def __init__(self, example):
        self.dt = 0.04
        self.x_dim = example.n_obs  # 变量个数
        self.u_dim = example.u_dim  # 控制维度
        self.D_zones = example.D_zones  # 不变区域
        self.I_zones = example.I_zones  # 初始区域
        self.G_zones = example.G_zones  # 目标区域
        self.U_zones = example.U_zones  # 非安全区域
        self.f = example.f  # 微分方程
        self.path = example.path
        self.u_bound = example.u
        self.R = example.R
        self.Q = example.Q
        self.constraint_dim = example.constraint_dim
        self.opti = ca.Opti()
        self.U = self.opti.variable(self.N, self.u_dim)  # 每一步控制器的参数
        self.x_0 = self.opti.parameter(self.x_dim).T  # 初始状态
        self.__add_subject()

    def get_next_state(self, x, u):

        dxdt = [x[i] + F(x, u) * self.dt for i, F in enumerate(self.f)]
        return ca.hcat(dxdt)

    def __add_subject(self):  # 添加约束
        self.x_final = self.opti.parameter(self.x_dim)

        obj = 0
        x_i = self.x_0
        for i in range(self.N):
            x_i = self.get_next_state(x_i, self.U[i, :])
            if self.D_zones.shape == 'box':  # 对不变式区域进行约束
                for j in range(self.constraint_dim):
                    self.opti.subject_to(self.opti.bounded(self.D_zones.low[j], x_i[j], self.D_zones.up[j]))
            else:
                center = self.D_zones.center[:self.constraint_dim].reshape(-1, self.constraint_dim)
                self.opti.subject_to(
                    self.opti.bounded(0, ca.sumsqr(x_i[:self.constraint_dim] - center), self.D_zones.r ** 2))

            if self.U_zones.shape == 'box':  # 对非安全区域进行约束
                if not self.U_zones.inner:
                    for j in range(self.constraint_dim):
                        self.opti.subject_to(self.opti.bounded(self.U_zones.low[j], x_i[j], self.U_zones.up[j]))
                else:
                    for j in range(self.constraint_dim):
                        self.opti.subject_to(self.opti.bounded(self.D_zones.low[j], x_i[j], self.U_zones.low[j]))
                        self.opti.subject_to(self.opti.bounded(self.U_zones.up[j], x_i[j], self.D_zones.up[j]))
            else:
                center = self.U_zones.center[:self.constraint_dim].reshape(-1, self.constraint_dim)
                if self.U_zones.inner:
                    self.opti.subject_to(
                        self.opti.bounded(self.U_zones.r + self.eps, ca.sumsqr(x_i[:self.constraint_dim] - center),
                                          np.inf))
                else:
                    self.opti.subject_to(
                        self.opti.bounded(0, ca.sumsqr(x_i[:self.constraint_dim] - center), self.U_zones.r + self.eps))

            obj = obj - ca.mtimes([(x_i - self.x_final.T), self.Q, (x_i - self.x_final.T).T]) + 1 * ca.mtimes(
                [self.U[i, :], self.R, self.U[i, :].T])  # 目标函数

        for i in range(self.u_dim):
            self.opti.subject_to(self.opti.bounded(-self.u_bound, self.U[:, i], self.u_bound))

        opts_setting = {'ipopt.max_iter': 400, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.minimize(obj)
        self.opti.solver('ipopt', opts_setting)

    def sample_init(self):
        """在初始区域内随机生成初始点"""
        if self.I_zones.shape == 'ball':
            # 在超球内进行采样：将正方体进行归一化，变成对单位球的表面采样，再对其半径进行采样。
            current_state = np.random.randn(self.x_dim)
            current_state = current_state / np.sqrt(sum(current_state ** 2)) * np.sqrt(
                self.I_zones.r) * np.random.random() ** (1 / self.x_dim)  # 此时球心在原点
            # random()^(1/d) 是为了均匀采样d维球体
            current_state += self.I_zones.center
            # theta = np.random.random() * 2 * np.pi
            # r = np.random.uniform(0, self.I_zones.r ** 2)
            # x = math.cos(theta) * (r ** 0.5) + self.I_zones.center[0]
            # y = math.sin(theta) * (r ** 0.5) + self.I_zones.center[1]
            # current_state =[x,y]
        else:
            current_state = np.array([np.random.random() - 0.5 for _ in range(self.x_dim)])
            current_state = current_state * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        return current_state

    def solve(self, current_state=None):
        # final_state =self.G_zones.center
        final_state = self.U_zones.center
        # print("init:",current_state)
        if current_state is None:
            current_state = self.sample_init()
            # print("init:",current_state)
        self.opti.set_value(self.x_final, final_state)
        time_out = 60
        count = 0
        state = [current_state]
        U_mpc = []
        while np.linalg.norm(current_state - final_state) > 1e-2 and count < 200:
            self.opti.set_value(self.x_0, current_state)  # 初始化初始位置
            sol = self.opti.solve()

            u = sol.value(self.U).reshape(self.N, -1)

            current_state = self.get_next_state(current_state, u[0, :])
            current_state = sol.value(current_state)

            state.append(current_state)
            U_mpc.append(u[0, :])
            count += 1
            print(current_state, u[0, :], count)

            d = np.sqrt(sum((current_state[:self.constraint_dim] - self.U_zones.center[:self.constraint_dim]) ** 2))
            if not (d <= self.U_zones.r) ^ self.U_zones.inner:
                print('进入不安全区域:距离为：', d)
                break
        print('error:', np.linalg.norm(current_state - final_state))
        print('step:', count)
        return np.array(state)[:-1], np.array(U_mpc)


# from Plot import plot
if __name__ == '__main__':
    ex = get_Env(13)
    mpc = MPC(ex)
    trace, U_mpc = mpc.solve()
    print(trace.shape, U_mpc.shape)
    # print(U_mpc)
    # state,action,next_state,reward,done
