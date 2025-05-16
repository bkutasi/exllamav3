import torch # For SamplingState potentially
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS

logger = logging.getLogger(__name__)

class SS_Smoothing(SS_Base):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def run(self, state: SamplingState):
        if self.factor == 0.0:
            if state.state == SS.INIT and state.in_logits is not None:
                if state.logits is None: state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return

        output_logits = None
        source_for_smoothing = None

        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Smoothing: state.in_logits is None for SS.INIT"); return
            source_for_smoothing = state.in_logits
        elif state.logits is not None:
            source_for_smoothing = state.logits
        elif state.in_logits is not None: # Fallback if state.logits is None but in_logits exists
            logger.warning("SS_Smoothing: state.logits is None, using state.in_logits as base for smoothing.")
            source_for_smoothing = state.in_logits
        else:
            logger.error(f"SS_Smoothing: No logits available (state.logits or state.in_logits is None) for state {state.state}. Cannot apply smoothing.")
            return

        if source_for_smoothing is state.in_logits:
            output_logits = source_for_smoothing.clone().float()
        else:
            output_logits = source_for_smoothing.float()

        V = output_logits.shape[-1]
        if V == 0:
            logger.error("SS_Smoothing: Vocab size is 0.")
            state.logits = output_logits
            state.state = SS.LOGITS
            return

        smoothing_value = self.factor / V
        output_logits = output_logits * (1.0 - self.factor) + smoothing_value

        state.logits = output_logits
        if state.state in [SS.LOGITS_S, SS.PROBS_S, SS.PROBS_N_S] :
            state.state = SS.LOGITS
            state.indices = None
        elif state.state in [SS.PROBS, SS.PROBS_N]:
             state.state = SS.LOGITS
        else:
            state.state = SS.LOGITS