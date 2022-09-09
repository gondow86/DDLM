from cmath import log
from math import exp
import torch
from scipy.misc import derivative
import numpy as np
import math

def init_dipole():
    return


def negative_distance_loss(gamma):
    """
    ・t-SNEで2次元に圧縮された値を座標として距離を計算する
    ・f_thetaXはFeature Extractorから出てきた値
    ・m_iN[i]はi個目の負極の位置（ランダムに初期化）

    """
    distance_i = abs(pow(f_thetaX - m_iN[i]))
    sum_p_yx = 0
    for k in range(n):
        distance_k = abs(pow(f_thetaX - m_iN[k]))
        sum_p_yx += exp(-1 * gamma * distance_k))
    p_yx = exp(-1 * gamma * distance_i) / sum_p_yx
    return -1 * log(p_yx)

def positive_distance_loss(gamma):
    return