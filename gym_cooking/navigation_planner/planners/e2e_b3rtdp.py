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

    def __init__(self, D, alpha, epsilon, beta, tau):
        """
        Initializes B3RTDP algorithm with its hyper-parameters.
        Refer to B3RTDP paper for how these hyper-parameters are used in their algorithm.

        https://www.researchgate.net/publication/364689660_B3RTDP_A_Belief_Branch_and_Bound_Real-Time_Dynamic_Programming_Approach_to_Solving_POMDPs

        Args:
            D: Belief discretization factor.
            alpha: Action convergence probability threshold.
            epsilon: Minimum value gap.
            beta: Minimum Convergence Frontier probability.
            tau: Trial termination ratio.
        """

        self.D = D
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.tau = tau

        raise NotImplementedError()

    def main(self, belief):
        """
        Before we start implementation, we'll want to determine our implementation of beliefs
        and determine how the algorithm will work for those.

        Then, we'll need to figure out how to model the beliefs within Bayesian Delegation.

        This function goes:

            1. InitializeCF(convergence_frontier c, beliefs)
            2. while not TERMINATECF(c)
            3.  b := SAMPLECF(c)
            4.  b^3rtdptrial(c)
            5.  updateCF(c)
            6. return greedy action
        """

        raise NotImplementedError()

    def b3rtdpTrial(self, convergence_frontier, belief):
        raise NotImplementedError()

    def terminateCF(self, convergence_frontier):
        raise NotImplementedError()

    def sampleCF(self, convergence_frontier):
        raise NotImplementedError()

    def updateCF(self, convergence_frontier):
        raise NotImplementedError()

    def get_action(self):
        raise NotImplementedError()
