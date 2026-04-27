import copy
import random

import numpy as np
import scipy as sp

NEG_INF_LOG_VAL = np.finfo(float).tiny


class SubtaskAllocDistribution:
    """Represents a distribution over subtask allocations."""

    def __init__(self, subtask_allocs):
        # subtask_allocs are a list of tuples of (subtask, subtask_agents).
        self.D = 200

        self.probs = {}
        if len(subtask_allocs) == 0:
            return

        self.discretized_probs = None

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
        new.D = self.D
        new.discretized_probs = self.discretized_probs
        return new

    def enumerate_subtask_allocs(self):
        return list(self.probs.keys())

    def get_list(self):
        return list(self.probs.items())

    def _ensure_discretized_probs(self):
        if self.discretized_probs is not None:
            return

        if len(self.probs) == 0:
            self.discretized_probs = {}
            return

        log_probs = list(self.probs.values())
        log_probs = [max(lp, NEG_INF_LOG_VAL) for lp in log_probs]
        probs = np.exp(log_probs)

        discretized = np.round(probs * self.D) / self.D
        total = float(np.sum(discretized))
        if total <= 0:
            discretized = np.full_like(discretized, 1.0 / len(discretized))
        else:
            discretized = discretized / total

        self.discretized_probs = {
            subtask_alloc: float(p)
            for subtask_alloc, p in zip(self.probs.keys(), discretized)
        }

    def get(self, subtask_alloc):
        self._ensure_discretized_probs()
        return self.discretized_probs[tuple(subtask_alloc)]

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
        self.discretized_probs = None

    def update(self, subtask_alloc, factor):
        self.probs[tuple(subtask_alloc)] += factor
        self.discretized_probs = None

    def delete(self, subtask_alloc):
        del self.probs[tuple(subtask_alloc)]
        self.discretized_probs = None

    def normalize(self):
        if len(self.probs) == 0:
            return self.probs

        log_probs = list(self.probs.values())
        log_total = sp.special.logsumexp(log_probs)

        for subtask_alloc in self.probs.keys():
            self.probs[subtask_alloc] -= log_total

        self.discretized_probs = None
        return self.probs
