import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage import *
from .affine import *
from .mux import *
from .gate import *

# The affine is a stage of the op now (see ``MuxAffine``): the operator
# applies it, and it reports itself.  ``Identity`` keeps ``preprocess`` and
# ``postprocess`` empty, so no affine is hiding in them any more.
#
# A composition can plug in a different affine by setting ``self.affine``
# before the operator's own setup runs:
#
#     class OpParamV5(TernaryMuxV4, Identity, SimpleGateClamp):
#         def __init__(self):
#             TernaryMuxV4.__init__(self)
#             Identity.__init__(self)
#             SimpleGateClamp.__init__(self)
#             self.affine = MyAffine()

# v0, has muxing competition during training
class OpParamV0(SoftmaxMux, Identity, SigmoidGate):
    def __init__(self):
        SoftmaxMux.__init__(self) # muxing parameters
        Identity.__init__(self) # blank
        SigmoidGate.__init__(self) # gate parameters

class OpParamV1(SigmoidMux, Identity, SigmoidGate):
    def __init__(self):
        SigmoidMux.__init__(self) # muxing parameters
        Identity.__init__(self) # blank
        SigmoidGate.__init__(self) # gate parameters

class OpParamV2(TernaryMuxV3, Identity, SigmoidGate):
    def __init__(self):
        TernaryMuxV3.__init__(self) # muxing parameters
        Identity.__init__(self) # blank
        SigmoidGate.__init__(self) # gate parameters


# class OpParamV3(TernaryMuxV4, Identity, SigmoidGate):
# class OpParamV3(TernaryMuxV4, Identity, SigmoidGateClamp):
class OpParamV3(TernaryMuxV4, Identity, SimpleGateClamp):
# class OpParamV3(TernaryMuxV4, Identity, SimpleGateClamp):


    def __init__(self):
        TernaryMuxV4.__init__(self) # muxing parameters
        # Identity.__init__(self) # blank
        Identity.__init__(self) # blank
        # SigmoidGateClamp.__init__(self) # gate parameters
        SimpleGateClamp.__init__(self) # gate parameters


# v4, like v3 but the mux weights vary over space
class OpParamV4(TernaryMuxV4Spatial, Identity, SimpleGateClamp):

    def __init__(self):
        TernaryMuxV4Spatial.__init__(self) # muxing parameters, over space
        Identity.__init__(self) # blank
        SimpleGateClamp.__init__(self) # gate parameters
