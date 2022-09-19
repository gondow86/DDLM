from cmath import log
from math import exp
import torch
from scipy.misc import derivative
import numpy as np
import math


def init_parameters():

    return


def negative_distance_loss(gamma):
    """
    ・t-SNEで2次元に圧縮された値を座標として距離を計算する
    ・f_thetaXはFeature Extractorから出てきた値
    ・m_N[i]はi個目の負極の位置（ランダムに初期化）
    ・m_N[y]はラベルyに対応する負極の位置
    """
    distance_i = abs(pow(f_thetaX - m_N[i]))
    sum_denominator = 0
    for k in range(n):
        distance_k = abs(pow(f_thetaX - m_N[k]))
        sum_denominator += exp(-1 * gamma * distance_k)
    p_yx = exp(-1 * gamma * distance_i) / sum_denominator
    l_cN = -1 * log(p_yx)

    # Regularization term
    l_oN = abs(pow(abs(pow(f_thetaX - m_N[y])) - s_y))

    l_N = l_cN + l_oN

    return l_N


def positive_distance_loss(beta):
    distance_i = abs(pow(f_thetaX - m_P[i]))
    sum_denominator = 0
    for k in range(len(m_P)):
        distance_k = abs(pow(f_thetaX - m_P[k]))
        sum_denominator += exp(beta * distance_i)
    p_yx = exp(-1 * beta * distance_i) / sum_denominator
    l_cP = -1 * log(p_yx)

    # Regularization term
    l_oP = abs(pow(abs(pow(f_thetaX - m_P[y])) - r_y))

    l_P = l_cP + l_oP

    return l_P
