from dataclasses import dataclass
from typing import Literal

from .wrappers import *
from .gravity_encoding import *
    

@dataclass
class EnvWrappersConfig:
    """Observation wrappers params"""
    
    num_obs_stacks: int = 1
    """number of times to stack observation"""

    positions: Literal['abs', 'rel', 'grav'] = 'abs'
    """positions: 
        - 'abs': absolute coordinates
        - 'rel': relative coordinates
        - 'grav': gradient gravity potential encoding (GravityEncoding)
    """
    
    statuses: Literal['no', 'ohe', 'cat'] = 'no'
    """add pedestrians statuses to obeservation as one-hot-encoded columns.
    NOTE: this value has no effect when `positions`='grad' is selected.
    """
    
    type: Literal['Dict', 'Box'] = 'Dict'
    """concatenate Dict-type observation to a Box-type observation
    (with added statuses to the observation)"""

    # ---- GravityEncoding params ----
    
    alpha: float = 3
    """alpha parameter of GravityEncoding. The value of alpha 
    determines the strength and shape of the potential function. 
    Higher value results in a stronger repulsion between the agent 
    and the pedestrians, a lower value results in a weaker repulsion.
    Typical expected values vary from 1 to 5."""

    # ---- post-init setup ----
    
    def __post_init__(self):
        # TODO add stacking wrappers
        assert self.num_obs_stacks == 1, NotImplementedError
        
    def wrap_env(self, env):
        """
        Possible options and how to get them:
        | pos:   | sta:  | type:  | how
        | - abs  | - no  | - dict | (which wrappers)
        | - rel  | - ohe | - box  | 
        | - grav | - cat |        | 
        |--------|-------|--------|-----
        |  abs   |  no   |  dict  | -
        |  abs   |  ohe  |  dict  | PedestriansStatuses(type='ohe')
        |  abs   |  cat  |  dict  | PedestriansStatuses(type='cat')
        |  abs   |  no   |  box   | MatrixObs(type='no')
        |  abs   |  ohe  |  box   | MatrixObs(type='ohe')
        |  abs   |  cat  |  box   | MatrixObs(type='cat')
        |  rel   |  no   |  dict  | RelativePosition()
        |  rel   |  ohe  |  dict  | RelativePosition() + PedestriansStatuses(type='ohe')
        |  rel   |  cat  |  dict  | RelativePosition() + PedestriansStatuses(type='cat')
        |  rel   |  no   |  box   | RelativePosition() + MatrixObs(type='no')
        |  rel   |  ohe  |  box   | RelativePosition() + MatrixObs(type='ohe')
        |  rel   |  cat  |  box   | RelativePosition() + MatrixObs(type='cat')
        |  grav  |  -    |  dict  | GravityEmbedding(alpha)
        |  grav  |  -    |  box   | TODO
        
        NOTE #1: `grav` position option utilizes information about state but 
        in its own way, so you don't need to add PedestriansStatuses() wrapper. 
        
        NOTE #2: to use Box version of `grav` position it is recommended to 
        just use `Flatten` observation wrapper from gymnasium.
        """
        
        if self.positions == 'grav':
            if self.type == 'Dict':
                return GravityEncoding(env, alpha=self.alpha)
            elif self.type == 'Box':
                raise NotImplementedError
            else:
                raise ValueError
        
        if self.positions == 'rel':
            env = RelativePosition(env)
        
        if self.type == 'Box':
            return MatrixObs(env, type=self.statuses)
    
        if self.statuses != 'no':
            env = PedestriansStatuses(env, type=self.statuses)
        
        return env