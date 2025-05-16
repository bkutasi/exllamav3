import torch # For SamplingState potentially
from typing import Optional, List # For SS_Base.prep type hint

from ..ss_definitions import SS_Base, SamplingState, SS

class SS_NoOp(SS_Base):
    def run(self, state: SamplingState):
        if state.state == SS.INIT and state.in_logits is not None and state.logits is None:
            state.logits = state.in_logits.float()
            state.state = SS.LOGITS
        pass