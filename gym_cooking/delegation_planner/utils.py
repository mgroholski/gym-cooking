import copy
import random

import numpy as np
import scipy as sp

NEG_INF_LOG_VAL = np.finfo(float).tiny


class SubtaskAllocDistribution:
    """Represents a distribution over subtask allocations."""

    def __init__(self, subtask_allocs):
        # subtask_allocs are a list of tuples of (subtask, subtask_agents).

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
            s += "{"
            for subtask, subtask_agent_names in subtask_alloc:
                s += f"({subtask}, {subtask_agent_names}),"

            s += "}: " + str(p) + "\n"
        return s

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.probs = copy.deepcopy(self.probs)
        return new

    def enumerate_subtask_allocs(self):
        return list(self.probs.keys())

    def get_list(self):
        return list(self.probs.items())

    def get(self, subtask_alloc):
        log_p = self.probs[tuple(subtask_alloc)]
        return np.exp(log_p)

    def get_max(self):
        if len(self.probs) > 0:
            max_prob = max(self.probs.values())
            max_subtask_allocs = [
                subtask_alloc
                for subtask_alloc, p in self.probs.items()
                if p == max_prob
            ]
            return random.choice(max_subtask_allocs)
        return None

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

    def get_entropy(self):
        entropy = 0
        for log_p in self.probs.values():
            p = np.exp(log_p)
            entropy += p * log_p

        return -entropy

    def get_max_entropy(self):
        n = len(self.probs)
        if not n:
            raise Exception("0 probs in distribution.")

        return np.log((1 / n))
