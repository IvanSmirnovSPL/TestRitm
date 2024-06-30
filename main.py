import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def minimize(functional: Callable, start_param: float, **kwargs) -> float:
    """Минимизация функционала.

    :param functional: минимизируемый функционал,
    :param start_param: начальное значение параметра.
    """
    k_max = 100  # макс. кол-во итераций
    tol = 1e-2  # невязка функционала
    delta = 1e-1  # доля
    param = start_param
    func = functional(param, **kwargs)
    k = 0
    while k < k_max or func > tol:
        f_left = functional(param * (1 - delta), **kwargs)
        f_right = functional(param * (1 + delta), **kwargs)
        grad = (f_right - f_left) / 2 / (delta * param)
        param -= 1e2 * grad
        func = functional(param, **kwargs)
        k += 1
    return param


def g(t: float) -> float:
    """Вид функции g из условия задачи

    :param t: время [сек].
    """
    return 9.81 + 5e-3 * np.sin(2 * np.pi * t)


def f(
    t: float, vals_old: NDArray, T_cur: float, m: float, L: float, g: Callable,
) -> NDArray:
    """Функция правых частей.

    :param t: время [сек],
    :param vals_old: предыдущее значение неизвестных,
    :param T_cur: текущее значение силы,
    :param m: масса груза [кг],
    :param L: длина стержня [м],
    :param g: функция g из условия.
    """

    assert vals_old.size == 4
    vals = np.zeros_like(vals_old)
    vals[0] = vals_old[1]
    vals[1] = - vals_old[0] * T_cur / m / L
    vals[2] = vals_old[3]
    vals[3] = - vals_old[2] * T_cur / m / L - g(t)
    return vals


def rk4(
    fun: Callable,
    t0: float,
    step: float,
    y0: NDArray,
    **kwargs,
) -> NDArray:
    """Рунге Кутта 4 порядка.

    :param fun: функция правых частей,
    :param t0: текущее время [сек],
    :param step: шаг по времени [сек],
    :param y0: текущее значение неизвестных.
    """
    k1 = step * fun(t0, y0, **kwargs)
    k2 = step * fun(t0 + step/2, y0 + k1/2, **kwargs)
    k3 = step * fun(t0 + step/2, y0 + k2/2, **kwargs)
    k4 = step * fun(t0 + step, y0 + k3, **kwargs)
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y0 + k


def func(t: float, vals_old: NDArray, step: float, **kwargs):
    """Формирование решения.

    :param t: текущее время [сек],
    :param t: текущее значение неизвестных,
    :param step: шаг по времени [сек].
    """

    def functional(T: float, **kwargs):
        """Функционал.

        :param T: сила.
        """
        v_ = rk4(fun=lambda t, v: f(t, v, T, **kwargs), t0=t, step=step, y0=vals_old)
        return abs(L**2 - (v_[0]**2 + v_[2]**2))

    T = minimize(functional, 1, **kwargs)  # находим текущее значение силы
    return rk4(  # решение на следующем шаге по времени
        fun=lambda _t, _v: f(_t, _v, T, **kwargs),
        t0=t,
        step=step,
        y0=vals_old,
    )


if __name__ == '__main__':
    res_path = Path(__file__).parent / 'result'
    if not res_path.exists():
        os.mkdir(res_path)

    vals_0 = [3, -1, -4, 1]  # [x_0, vx_0, y, y_0], СИ
    m = 1  # кг
    L = 5  # м
    T = 100  # расчёт в диапазоне [0, T], сек

    N = int(100 * T + 1)  # кол-во точек по времени
    t_moments = np.linspace(0, T, num=N)
    step = T / (N - 1)
    sol = np.zeros((t_moments.size, 4))
    for i in tqdm(range(t_moments.size)):
        if i == 0:
            sol[i, :] = vals_0
        else:
            sol[i, :] = func(t_moments[i - 1], sol[i - 1, :], step=step, m=m, L=L, g=g)


    fig = plt.figure(figsize=(10, 15))
    ax = fig.subplots(2, 1)
    ax[0].scatter(0, 0, c='k')
    ax[0].scatter(vals_0[0], vals_0[2], lw=5, c='r')
    ax[0].plot(sol[:, 0], sol[:, 2], '-o', c='b', label='Положение груза от времени')
    ax[0].set_xlabel('x, м')
    ax[0].set_xlabel('y, м')
    ax[0].grid(True)

    ax[1].plot(np.sqrt(sol[:, 0]**2 + sol[:, 2]**2))
    ax[1].grid(True)
    ax[1].set_ylabel('Длина маятника, м')
    ax[1].set_xlabel('Итерации')
    plt.savefig(res_path / f'{m=}_{L=}_{T=}.png')
