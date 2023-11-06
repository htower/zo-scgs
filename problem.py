# Copyright (c) 2023 - present
#
# Anton Anikin <anton@anikin.xyz>
# Alexander Gasnikov <gasnikov.av@mipt.ru>
# Aleksandr Lobanov <lobanov.av@mipt.ru>
# Alexander Gornov <gornov@icc.ru>
# Sergey Chukanov <chukanov47@mail.ru>
#

import numpy as np


class Problem:
    def __init__(self, size: int) -> None:
        self.A = np.random.rand(size, size)
        self.A = np.matmul(self.A, self.A.T)

        self.x_star = np.random.rand(size, 1)
        self.x_star /= self.x_star.sum()

        self.b = np.matmul(self.A, self.x_star)

        f = np.matmul(self.A, self.x_star) - 2 * self.b
        f = np.dot(np.squeeze(self.x_star), np.squeeze(f))
        f *= 0.5
        self.f_star = f

        self.sigma = 0.0

        self.f_cnt = 0

        self.df_type = "normal"

        # =====================================================================

        self.size = self.d = self.n = size
        self.D = 2.0

        self.M = self.M1 = abs(self.A).max()

        w, _ = np.linalg.eigh(self.A)
        self.M2 = w[-1]

        # smooth case
        self.L = self.M1

        # non-smooth case
        # self.L = np.sqrt(self.d) * self.M1 * self.M2 / eps

        print("problem:")
        print(f"     n = {self.n}")
        print(f"     D = {self.D}")
        print(f"L = M1 = {self.M1:e}")
        print(f"    M2 = {self.M2:e}")
        print(f"    f* = {self.f_star}")
        print()

        self.idx = 0

    def f(self, x: np.ndarray) -> float:
        assert x.shape == (self.size, 1)

        # 1/2 xAx - bx -> min
        # 1/2x(Ax - 2b) -> min
        #    x(Ax - 2b) -> min
        axm2b = np.matmul(self.A, x) - 2 * self.b
        f = 0.5 * np.dot(np.squeeze(x), np.squeeze(axm2b))
        f -= self.f_star

        self.f_cnt += 1

        return f

    def f_ksi(self, x: np.ndarray, ksi: int) -> float:
        assert x.shape == (self.size, 1)
        assert ksi >= 0
        assert ksi < self.size

        f = 0.5 * np.dot(self.A[ksi], np.squeeze(x)) - self.b[ksi]
        self.f_cnt += 1

        return f

    def f_ksi2(self, ksi: int, eta: int) -> float:
        assert ksi >= 0
        assert ksi < self.size

        assert eta >= 0
        assert eta < self.size

        f = 0.5 * self.A[ksi, eta] - self.b[eta]
        self.f_cnt += 1

        return f

    def df(self, x: np.ndarray, batch_size: int, step: float):
        assert batch_size > 0
        assert step > 0.0

        g = np.zeros_like(x)

        if self.df_type == "coord":
            # f0 = self.f(x)
            for i in range(self.size):
                u_i = self.standard_basis_vector(i)

                g_i = self.f(x + step * u_i) - self.f(x - step * u_i)
                g_i /= 2.0 * step

                # g_i = self.f(x + step * u_i) - f0
                # g_i /= step

                g += g_i * u_i

            return g

        for i in range(batch_size):
            if self.df_type == "normal":  # ZSCG
                u_i = self.sample_norml()

                # ksi_i = self.sample_ksi(x)
                # g_i = self.f_ksi(x + step * u_i, ksi_i) - self.f_ksi(x, ksi_i)

                g_i = self.f(x + step * u_i) - self.f(x)

                g_i /= step

            elif self.df_type == "spherical":  # own method
                u_i = self.sample_spherical()

                # ksi_i = self.sample_ksi(x)
                # g_i = self.f_ksi(x + step * u_i, ksi_i) - self.f_ksi(x - step * u_i, ksi_i)

                g_i = self.f(x + step * u_i) - self.f(x - step * u_i)

                g_i /= 2.0 * step
                g_i *= self.d

            else:
                u_i = self.sample_spherical()

                z = x + step * u_i
                ksi_i = self.sample_ksi(z)
                eta_i = self.sample_ksi(z)

                z = x - step * u_i
                ksi_i2 = self.sample_ksi(z)
                eta_i2 = self.sample_ksi(z)

                g_i = self.f_ksi2(ksi_i, eta_i) - self.f_ksi2(ksi_i2, eta_i2)
                g_i /= 2.0 * step
                g_i *= self.d

            g += g_i * u_i
        g /= batch_size

        return g

    def sample_ksi(self, x: np.ndarray) -> int:
        # non-uniform distribution controlled by the values of 'x'
        z = x / np.linalg.norm(x)
        i_sort = np.argsort(np.squeeze(z))
        p = np.random.rand()  # [0, 1)

        sum = 0.0
        for i in i_sort:
            sum += z[i]
            if sum >= p:
                return i

        return i_sort[-1]

    def sample_spherical(self) -> np.ndarray:
        v = np.random.randn(self.size, 1)
        v /= np.linalg.norm(v)

        return v

    def sample_norml(self) -> np.ndarray:
        v = np.random.randn(self.size, 1)

        return v

    def standard_basis_vector(self, index) -> np.ndarray:
        assert index >= 0
        assert index < self.size

        v = np.zeros((self.size, 1))
        v[index] = 1.0

        return v
