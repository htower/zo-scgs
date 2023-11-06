#!/usr/bin/env python3
#
# Copyright (c) 2023 - present
#
# Anton Anikin <anton@anikin.xyz>
# Alexander Gasnikov <gasnikov.av@mipt.ru>
# Aleksandr Lobanov <lobanov.av@mipt.ru>
# Alexander Gornov <gornov@icc.ru>
# Sergey Chukanov <chukanov47@mail.ru>
#

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from methods import ZO_SCGS, ZSCG, Global
from problem import Problem


def main():
    seed = 0
    np.random.seed(seed)

    problem = Problem(100)

    # noise level
    problem.sigma = 1e-4

    # x0 = np.ones((problem.size, 1))
    # x0 /= x0.sum()

    # x0 = np.random.rand(problem.size, 1)
    # x0 /= x0.sum()

    x0 = np.zeros((problem.size, 1))
    x0[0] = 1.0

    max_f_cnt = int(5e6)
    max_iters = int(1e6)
    Global.BK = 50  # 50 * (problem size) == 5000

    zscg = ZSCG()
    zscg.max_iters = max_iters
    zscg.max_f_cnt = max_f_cnt

    zo_scgs = ZO_SCGS()
    zo_scgs.smooth = True
    zo_scgs.eps = 1e-6
    zo_scgs.max_iters = max_iters
    zo_scgs.max_f_cnt = max_f_cnt

    problem.df_type = "normal"
    # problem.df_type = "spherical"
    # problem.df_type = "coord"
    zscg.run(problem, x0, seed)

    # problem.df_type = "normal"
    problem.df_type = "spherical"
    # problem.df_type = "ksi2"
    # problem.df_type = "coord"
    zo_scgs.run(problem, x0, seed)

    zscg.df.to_csv("zscg.csv", sep=";", header=True, index=False)
    zo_scgs.df.to_csv("zo_scgs.csv", sep=";", header=True, index=False)

    # ==========================================================================

    font = {
        "family": "sans",
        # "weight" : "bold",
        "size": 24,
    }
    matplotlib.rc("font", **font)

    labels = {
        "iter": "iterations",
        "f_cnt": "gradient-free oracle calls",
        "f_i": "$f(x_k) - f^*$",
    }

    y_var = "f_i"
    for x_var in ["iter", "f_cnt"]:
        # fix for log-scale
        zo_scgs.df[x_var] += 1
        zscg.df[x_var] += 1

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 10, forward=True)

        ax.plot(zscg.df[x_var], zscg.df[y_var], label=zscg.df["method"][0])
        ax.plot(zo_scgs.df[x_var], zo_scgs.df[y_var], label=zo_scgs.df["method"][0])

        plt.xlabel(labels[x_var])
        plt.ylabel(labels[y_var])

        plt.xscale("log")
        plt.yscale("log")
        plt.legend()

        plt.savefig(f"{x_var}_out.png")

    # plt.show()


if __name__ == "__main__":
    main()
