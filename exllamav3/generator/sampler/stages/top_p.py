import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS
from .normalize import SS_Normalize # For prep() method
from .sort import SS_Sort # For prep() method
from .noop import SS_NoOp # For alt() method

logger = logging.getLogger(__name__)

class SS_TopP(SS_Base):
    def __init__(self, top_p: float):
        self.top_p = top_p
        if not (0.0 <= top_p <= 1.0):
            self.top_p = max(0.0, min(1.0, top_p))

    def run(self, state: SamplingState):
        if self.top_p >= 0.9999999:
            if state.state == SS.PROBS_N_S:
                state.state = SS.PROBS_S
            return

        if state.state != SS.PROBS_N_S:
            logger.error(f"SS_TopP: Invalid input state {state.state}. Requires sorted, normalized probabilities (PROBS_N_S).")
            if state.logits is not None or state.probs is not None or state.in_logits is not None:
                logger.warning("SS_TopP: Attempting recovery by normalizing and sorting.")
                SS_Normalize().run(state)
                SS_Sort().run(state)
                if state.state != SS.PROBS_N_S:
                    logger.error(f"SS_TopP: Recovery to PROBS_N_S failed. Current state: {state.state}")
                    raise ValueError("SS_TopP requires PROBS_N_S. Auto-recovery failed.")
            else:
                raise ValueError("SS_TopP requires PROBS_N_S and no data for recovery.")

        if state.probs is None or state.indices is None:
            logger.error("SS_TopP: state.probs or state.indices is None for PROBS_N_S"); return

        probs_to_filter = state.probs

        cumsum_probs = torch.cumsum(probs_to_filter, dim=-1)

        mask_out = torch.zeros_like(probs_to_filter, dtype=torch.bool)
        if probs_to_filter.shape[-1] > 1:
            mask_out[..., 1:] = cumsum_probs[..., :-1] > self.top_p

        probs_to_filter[mask_out] = 0.0

        state.state = SS.PROBS_S

    def prep(self, in_state: SS) -> Optional[List[type[SS_Base]]]:
        prep_steps_classes: List[type[SS_Base]] = []

        if in_state == SS.LOGITS_S:
            prep_steps_classes.append(SS_Normalize)
        elif in_state == SS.PROBS_S:
             prep_steps_classes.append(SS_Normalize)
        elif in_state == SS.LOGITS:
            prep_steps_classes.append(SS_Normalize)
            prep_steps_classes.append(SS_Sort)
        elif in_state == SS.PROBS_N:
            prep_steps_classes.append(SS_Sort)
        elif in_state == SS.PROBS:
            prep_steps_classes.append(SS_Sort)
            prep_steps_classes.append(SS_Normalize)
        elif in_state == SS.INIT:
            prep_steps_classes.append(SS_Normalize)
            prep_steps_classes.append(SS_Sort)

        return prep_steps_classes if prep_steps_classes else None


    def alt(self):
        if self.top_p >= 0.9999999:
            return SS_NoOp()
        return None