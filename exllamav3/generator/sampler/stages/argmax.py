import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS

logger = logging.getLogger(__name__)

class SS_Argmax(SS_Base):
    def run(self, state: SamplingState):
        if state.sample is not None:
            state.state = SS.DONE
            return

        source_tensor = None
        is_sorted_input = False

        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Argmax: state.in_logits is None for SS.INIT"); return
            source_tensor = state.in_logits
        elif state.state == SS.LOGITS:
            if state.logits is None: logger.error("SS_Argmax: state.logits is None for SS.LOGITS"); return
            source_tensor = state.logits
        elif state.state == SS.LOGITS_S:
            if state.logits is None: logger.error("SS_Argmax: state.logits is None for SS.LOGITS_S"); return
            source_tensor = state.logits
            is_sorted_input = True
        elif state.state in [SS.PROBS, SS.PROBS_N]:
            if state.probs is None: logger.error(f"SS_Argmax: state.probs is None for {state.state}"); return
            source_tensor = state.probs
        elif state.state in [SS.PROBS_S, SS.PROBS_N_S]:
            if state.probs is None: logger.error(f"SS_Argmax: state.probs is None for {state.state}"); return
            source_tensor = state.probs
            is_sorted_input = True
        else:
            logger.error(f"SS_Argmax: Invalid input state {state.state}")
            return

        if is_sorted_input:
            if state.indices is None: logger.error(f"SS_Argmax: state.indices is None for sorted state {state.state}"); return
            temp_argmax_indices = torch.argmax(source_tensor, dim = -1, keepdim=True)
            state.sample = state.indices.gather(dim=-1, index=temp_argmax_indices).squeeze(-1)
        else:
            state.sample = torch.argmax(source_tensor, dim = -1)
        
        state.state = SS.DONE