# Copyright (c) 2023 - present
#
# Anton Anikin <anton@anikin.xyz>
# Alexander Gasnikov <gasnikov.av@mipt.ru>
# Aleksandr Lobanov <lobanov.av@mipt.ru>
# Alexander Gornov <gornov@icc.ru>
# Sergey Chukanov <chukanov47@mail.ru>
#

from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from problem import Problem


# global variables
class Global:
    BK = None


class Method(metaclass=ABCMeta):
    def __init__(self, name: str) -> None:
        assert name != ""

        self.name = name
        self._nl = max(len(name), len("method"))

        self.max_iters = 50
        self.max_f_cnt = 1e6

        self.df = pd.DataFrame()

    def run(self, problem: Problem, x0: np.ndarray, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        problem.f_cnt = 0

        self._log_start(problem, x0)
        self._run(problem, x0)

    def _log(self, problem: Problem, iter: int, f_i: float, batch: int, df_h: float):
        end = "\n"
        end = "\r"
        print(
            f"{self.name:{self._nl}} {iter:4} {f_i: e} {problem.f_cnt:8} {batch:7} {df_h: e}",
            end=end,
        )

        row = {}
        row["method"] = [self.name]
        row["iter"] = [iter]
        row["f_i"] = [f_i]
        row["f_cnt"] = [problem.f_cnt]
        row["batch"] = [batch]
        row["df_h"] = [df_h]

        self.df = pd.concat([self.df, pd.DataFrame.from_dict(row)], ignore_index=True)

    def _log_start(self, problem: Problem, x0: np.ndarray):
        print(
            f"{'method':>{self._nl}} {'iter':4} {' f_i':13} {'f_cnt':>8} {'batch':>7} {' df_h':13}"
        )

        self.df = pd.DataFrame()

        f0 = problem.f(x0)
        self._log(problem, 0, f0, 0, 0.0)
        print()

    @abstractmethod
    def _run(self, problem: Problem, x0: np.ndarray) -> np.ndarray:
        pass


class ZSCG(Method):
    def __init__(self) -> None:
        super().__init__("ZSCG")

    def _run(self, problem: Problem, x0: np.ndarray) -> np.ndarray:
        x_i = x0.copy()
        y = np.zeros_like(x0)

        n = problem.size
        D = problem.D

        # xx = np.zeros_like(x0)

        iter = 0
        for t in range(self.max_iters):
            iter += 1

            if problem.f_cnt >= self.max_f_cnt:
                break

            batch_size = (n + 5) * (t + 2) ** 2

            if Global.BK is not None:
                batch_size = n * Global.BK
                batch_size = np.ceil(batch_size)
                batch_size = int(batch_size)

            df_h = D
            df_h /= (n + 5) ** (3 / 2) * (t + 2)

            g = problem.df(x_i, batch_size, df_h)

            # argmin_{v \in Q} <g, v>, where Q -- unit simplex
            y.fill(0.0)
            y[g.argmin()] = 1.0

            gamma = 2 / (t + 2)
            # x_i = (1.0 - gamma) * x_i + gamma * y
            x_i += gamma * (y - x_i)

            f_i = problem.f(x_i)

            # xx += x_i
            # f_i = problem.f(xx / iter)

            self._log(problem, iter, f_i, batch_size, df_h)

        print()
        print()

        return x_i


class ZO_SCGS(Method):
    def __init__(self) -> None:
        super().__init__("ZO-SCGS")

        self.smooth = True
        self.eps = 1e-6

    def _run(self, problem: Problem, x0: np.ndarray) -> np.ndarray:
        x_i = x0.copy()
        y_i = x0.copy()

        d = problem.d
        D = problem.D
        M = problem.M
        M2 = problem.M2

        L = problem.L
        print(f"L = {L:e}")
        if self.smooth is False:
            L = np.sqrt(d) * M * M2 / self.eps
            print(f"L = {L:e}")

        # FIXME
        gamma = self.eps
        # gamma /= 2 * M2

        # xx = np.zeros_like(x0)

        k = 0
        for t in range(self.max_iters):
            k += 1

            if problem.f_cnt >= self.max_f_cnt:
                break

            dzeta = 3 / (k + 3)
            z_i = (1 - dzeta) * x_i + dzeta * y_i

            eta = 4 * L / (t + 3)
            beta = L * D**2
            beta /= (t + 1) * (t + 2)

            sigma2 = 2 * np.sqrt(2) * np.log(d) * M2**2
            sigma2 = np.log(d)

            batch_size = sigma2 * (t + 3) ** 3
            batch_size /= (L * D) ** 2

            batch_size = np.ceil(batch_size)
            batch_size = int(batch_size)

            if Global.BK is not None:
                batch_size = d * Global.BK
                batch_size = np.ceil(batch_size)
                batch_size = int(batch_size)

            g = problem.df(z_i, batch_size, gamma)

            # argmin_{v \in Q} <g, v>, where Q -- unit simplex
            y_i = self._cg(g, y_i, eta, beta)

            x_i = (1.0 - dzeta) * x_i + dzeta * y_i
            # x_i += dzeta * (y_i - x_i)

            f_i = problem.f(x_i)

            # xx += x_i
            # f_i = problem.f(xx / k)

            self._log(problem, k, f_i, batch_size, gamma)

        print()
        print()

        return x_i

    def _cg(
        self, g0: np.ndarray, u0: np.ndarray, eta: float, beta: float
    ) -> np.ndarray:
        g = g0.copy()
        u = u0.copy()
        v = np.zeros_like(g0)

        iter = 0
        while True:
            iter += 1

            # argmin_{v \in Q} <g, v>, where Q -- unit simplex
            v.fill(0.0)
            v[g.argmin()] = 1.0

            delta = np.squeeze(u - v)
            alpha = np.dot(np.squeeze(g), delta)

            if (alpha <= beta) and (iter > 1):
                return u

            alpha /= eta * np.dot(delta, delta)
            alpha = min(alpha, 1.0)

            u = u + alpha * (v - u)
            # u -= alpha * delta

            g = g0 + eta * (u - u0)
