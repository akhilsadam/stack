import torch
import torch.nn as nn
import torch.nn.functional as F

from .mux import *
from .affine import *
from .gate import *

# v0, has muxing competition during training
class OpParamV0(SoftmaxMux, PostAffine, SigmoidGate):
    def __init__(self):
        SoftmaxMux.__init__(self) # muxing parameters
        PostAffine.__init__(self) # affine layer, after muxing
        SigmoidGate.__init__(self) # gate parameters

class OpParamV1(SigmoidMux, PostAffine, SigmoidGate):
    def __init__(self):
        SigmoidMux.__init__(self) # muxing parameters
        PostAffine.__init__(self) # affine layer, after muxing
        SigmoidGate.__init__(self) # gate parameters

class OpParamV2(TernaryMux, Identity, SigmoidGate):
    def __init__(self):
        # TernaryMux.__init__(self) # muxing parameters
        # PostAffine.__init__(self, aff=AffineReLU) # affine layer, after muxing

        TernaryMux.__init__(self) # muxing parameters
        PostAffine.__init__(self) # affine layer, after muxing
        SigmoidGate.__init__(self) # gate parameters

        # TernaryMux.__init__(self) # muxing parameters
        # Identity.__init__(self) # (no) affine layer, after muxing
        # SigmoidGate.__init__(self) # gate parameters