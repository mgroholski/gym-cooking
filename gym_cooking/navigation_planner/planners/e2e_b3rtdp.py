from enum import Enum

import numpy as np


class PlannerLevel(Enum):
    LEVEL1 = 1
    LEVEL0 = 0


def argmin(vector):
    e_x = np.array(vector) == min(vector)

    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]


def argmax(vector):
    e_x = np.array(vector) == max(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]


class E2E_B3RTDP:
    """
    Belief Branch and Bound Real-Time Dynamic Programming Algorithm

    Details on this algorithm: https://www.researchgate.net/publication/364689660_B3RTDP_A_Belief_Branch_and_Bound_Real-Time_Dynamic_Programming_Approach_to_Solving_POMDPs
    """

    def __init__(self):
        raise NotImplementedError()
