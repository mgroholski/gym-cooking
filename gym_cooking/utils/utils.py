def agent_settings(arglist, agent_name):
    if agent_name[-1] == "1":
        return arglist.model1
    elif agent_name[-1] == "2":
        return arglist.model2
    elif agent_name[-1] == "3":
        return arglist.model3
    elif agent_name[-1] == "4":
        return arglist.model4
    else:
        raise ValueError("Agent name doesn't follow the right naming, `agent-<int>`")


class AgentBelief:
    def __init__(self):
        """
        Within this class we'll want to hold beliefs about the other agent's capabilities and the current incomplete subtask set.

        """

        raise NotImplementedError()

    def update(self, obs):
        raise NotImplementedError()

    def is_agent_capable(self, subtask):
        raise NotImplementedError()

    def get_incomplete_subtasks(self):
        raise NotImplementedError()
