import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

# SS_NoOp is used in the alt() method.
# Assume SS_Base, SamplingState, SS, and SS_NoOp are in ss_definitions.py for now.
from ..ss_definitions import SS_Base, SamplingState, SS
from .noop import SS_NoOp # Import SS_NoOp from its new sibling module

logger = logging.getLogger(__name__)

class SS_Skew(SS_Base):
    def __init__(self, skew: float):
        super().__init__()
        self.skew = skew

    def run(self, state: SamplingState):
        if abs(self.skew) < 1e-9 : 
            if state.state == SS.INIT and state.in_logits is not None:
                if state.logits is None: 
                    state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return

        initial_state_had_indices = state.state in [SS.LOGITS_S, SS.PROBS_S, SS.PROBS_N_S]
        current_probs_tensor = None 
        target_dtype = torch.float 

        source_tensor_for_softmax = None
        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Skew: state.in_logits is None for SS.INIT"); return
            source_tensor_for_softmax = state.in_logits
            if state.logits is None: state.logits = state.in_logits.to(target_dtype) 
        elif state.state in [SS.LOGITS, SS.LOGITS_S]:
            if state.logits is None: logger.error(f"SS_Skew: state.logits is None for {state.state}"); return
            source_tensor_for_softmax = state.logits
        elif state.state in [SS.PROBS, SS.PROBS_N, SS.PROBS_S, SS.PROBS_N_S]:
            if state.probs is None: logger.error(f"SS_Skew: state.probs is None for {state.state}"); return
            current_probs_tensor = state.probs.to(target_dtype) 
        else:
            logger.error(f"SS_Skew: Invalid input state {state.state}")
            return

        if current_probs_tensor is None and source_tensor_for_softmax is not None:
            current_probs_tensor = torch.softmax(source_tensor_for_softmax.to(target_dtype), dim=-1)
        elif current_probs_tensor is None:
            logger.error("SS_Skew: Failed to derive current_probs_tensor."); return

        current_probs_tensor.pow_(1.0 / (1.0 + self.skew))
        state.probs = current_probs_tensor 

        if initial_state_had_indices: 
            state.state = SS.PROBS_S 
        else:
            state.state = SS.PROBS 
        if state.logits is not None:
            pass 

    def alt(self):
        if abs(self.skew) < 1e-9:
            return SS_NoOp()
        return None