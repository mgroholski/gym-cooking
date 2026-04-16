import copy
import random
from collections import namedtuple

import numpy as np
import scipy as sp
from utils.utils import agent_settings

NEG_INF_LOG_VAL = -700


class SubtaskAllocDistribution:
    """Represents a distribution over subtask allocations."""

    def __init__(self, subtask_allocs):
        # subtask_allocs are a list of tuples of (subtask, subtask_agents).
        self.D = 50

        self.probs = {}
        if len(subtask_allocs) == 0:
            return

        prior = np.log(1.0 / (len(subtask_allocs)))
        print("set prior", prior)

        for subtask_alloc in subtask_allocs:
            self.probs[tuple(subtask_alloc)] = prior

    def __str__(self):
        s = ""
        for subtask_alloc, p in self.probs.items():
            s + "{"
            for subtask, subtask_agent_names in subtask_alloc:
                s += f"({subtask}, {subtask_agent_names}),"

            s += "}: " + str(p) + "\n"
        return s

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.probs = copy.deepcopy(self.probs)
        new.D = self.D
        return new

    def enumerate_subtask_allocs(self):
        return list(self.probs.keys())

    def get_list(self):
        return list(self.probs.items())

    def get(self, subtask_alloc):
        log_p = self.probs[tuple(subtask_alloc)]
        log_p = max(log_p, NEG_INF_LOG_VAL)
        p = np.exp(log_p)

        # Discretize to a factor of D
        p_discrete = np.round(p * self.D) / float(self.D)
        return float(np.clip(p_discrete, 0.0, 1.0))

    def get_max(self):
        try:
            if len(self.probs) > 0:
                max_prob = max(self.probs.values())
                max_subtask_allocs = [
                    subtask_alloc
                    for subtask_alloc, p in self.probs.items()
                    if p == max_prob
                ]
                return random.choice(max_subtask_allocs)
            return None
        except Exception as e:
            print(e)
            breakpoint()

    def get_max_bucketed(self):
        subtasks = []
        probs = []
        for subtask_alloc, p in self.probs.items():
            for t in subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    # If already accounted for, then add probability.
                    if t in subtasks:
                        probs[subtasks.index(t)] += p
                    # Otherwise, make a new element in the distribution.
                    else:
                        subtasks.append(t)
                        probs.append(p)
        best_subtask = subtasks[np.argmax(probs)]
        return self.probs.get_best_containing(best_subtask)

    def get_best_containing(self, subtask):
        """Return max likelihood subtask_alloc that contains the given subtask."""
        valid_subtask_allocs = []
        valid_p = []
        for subtask_alloc, p in self.probs.items():
            if subtask in subtask_alloc:
                valid_subtask_allocs.append(subtask)
                valid_p.append(p)
        return valid_subtask_allocs[np.argmax(valid_p)]

    def set(self, subtask_alloc, value):
        self.probs[tuple(subtask_alloc)] = value

    def update(self, subtask_alloc, factor):
        self.probs[tuple(subtask_alloc)] += factor

    def delete(self, subtask_alloc):
        del self.probs[tuple(subtask_alloc)]

    def normalize(self):
        if len(self.probs) == 0:
            return self.probs

        log_probs = list(self.probs.values())
        log_total = sp.special.logsumexp(log_probs)

        for subtask_alloc in self.probs.keys():
            self.probs[subtask_alloc] -= log_total

        return self.probs
