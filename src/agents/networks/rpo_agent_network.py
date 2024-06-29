from dataclasses import dataclass

class RPOAgentNetworkConfig:

    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""


class RPOAgentNetwork:
    ...