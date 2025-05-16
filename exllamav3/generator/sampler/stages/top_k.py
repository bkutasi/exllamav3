import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS
from .sort import SS_Sort # For prep() and run() logic
from .noop import SS_NoOp # For alt() method

logger = logging.getLogger(__name__)

class SS_TopK(SS_Base):
    def __init__(self, top_k: int):
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except ValueError:
                raise ValueError(f"top_k value must be convertible to an integer, got {top_k}")
        self.top_k = top_k


    def run(self, state: SamplingState):
        if self.top_k <= 0:
            if state.state == SS.PROBS_N_S: state.state = SS.PROBS_S
            return

        current_data_tensor = None
        is_logits_input = False
        is_valid_sorted_state = False

        if state.state == SS.LOGITS_S:
            if state.logits is not None and state.indices is not None:
                current_data_tensor = state.logits
                is_logits_input = True
                is_valid_sorted_state = True
            else: logger.error("SS_TopK: state.logits or state.indices is None for LOGITS_S"); return
        elif state.state in [SS.PROBS_S, SS.PROBS_N_S]:
            if state.probs is not None and state.indices is not None:
                current_data_tensor = state.probs
                is_valid_sorted_state = True
            else: logger.error(f"SS_TopK: state.probs or state.indices is None for {state.state}"); return

        if not is_valid_sorted_state:
            logger.error(f"SS_TopK: Invalid input state {state.state} or missing data. Requires sorted logits or probabilities with indices.")
            if state.logits is not None or state.probs is not None or state.in_logits is not None:
                logger.warning("SS_TopK: Attempting recovery by sorting.")
                SS_Sort().run(state)
                if state.state == SS.LOGITS_S and state.logits is not None:
                    current_data_tensor = state.logits
                    is_logits_input = True
                elif state.state in [SS.PROBS_S, SS.PROBS_N_S] and state.probs is not None:
                    current_data_tensor = state.probs
                else:
                    logger.error("SS_TopK: Recovery sort failed to produce a usable sorted state.")
                    raise ValueError("SS_TopK requires sorted input. Auto-sort recovery failed.")
            else:
                raise ValueError("SS_TopK requires sorted input and no data available to attempt recovery sort.")

        if current_data_tensor is None:
            logger.error("SS_TopK: current_data_tensor is None before filtering. Aborting TopK."); return

        vocab_size = current_data_tensor.shape[-1]
        actual_k = min(self.top_k, vocab_size)

        if actual_k >= vocab_size:
            if state.state == SS.PROBS_N_S: state.state = SS.PROBS_S
            return

        if is_logits_input:
            current_data_tensor[..., actual_k:] = -float("inf")
        else:
            current_data_tensor[..., actual_k:] = 0.0
            state.state = SS.PROBS_S

    def prep(self, in_state: SS) -> Optional[List[type['SS_Base']]]:
        if in_state not in [SS.LOGITS_S, SS.PROBS_S, SS.PROBS_N_S]:
            return [SS_Sort]
        return None

    def alt(self):
        if self.top_k <= 0:
            return SS_NoOp()
        return None