import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from .model_util import recursive_clone
from .base.base_model import BaseModel
from .model_arch import Network

def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)

class EventHDR(BaseModel):
    def __init__(self, kwargs):
        super().__init__()
        self.num_bins = kwargs['num_in_ch']  # legacy
        self.model = Network(**kwargs)
    
    @property
    def states(self):
        return copy_states(self.model.states)

    @states.setter
    def states(self, state):
        self.model.state = state

    def reset_states(self):
        self.model.state = None

    def forward(self, event_tensor):
        output = self.model(event_tensor)
        return {'image': output}
