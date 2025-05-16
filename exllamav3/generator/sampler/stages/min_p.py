import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS
from .normalize import SS_Normalize # For prep() method
from .noop import SS_NoOp # For alt() method

logger = logging.getLogger(__name__)

class SS_MinP(SS_Base):
    def __init__(self, min_p: float):
        self.min_p = min_p
        if not (0.0 <= min_p <= 1.0):
            self.min_p = max(0.0, min(1.0, min_p))

    def run(self, state: SamplingState):
        if abs(self.min_p) < 1e-9 :
            if state.state == SS.PROBS_N: state.state = SS.PROBS
            elif state.state == SS.PROBS_N_S: state.state = SS.PROBS_S
            return

        if state.state not in [SS.PROBS_N, SS.PROBS_N_S]:
            logger.error(f"SS_MinP: Invalid input state {state.state}. Requires normalized probabilities (PROBS_N or PROBS_N_S).")
            if state.logits is not None or state.probs is not None or state.in_logits is not None:
                logger.warning("SS_MinP: Attempting recovery by normalizing.")
                SS_Normalize().run(state)
                if state.state not in [SS.PROBS_N, SS.PROBS_N_S]:
                    logger.error(f"SS_MinP: Recovery to PROBS_N/PROBS_N_S failed. Current state: {state.state}")
                    raise ValueError("SS_MinP requires PROBS_N or PROBS_N_S. Auto-recovery failed.")
            else:
                raise ValueError("SS_MinP requires PROBS_N or PROBS_N_S and no data for recovery.")


        if state.probs is None:
            logger.error(f"SS_MinP: state.probs is None for state {state.state}"); return

        probs_to_filter = state.probs.float()
        if state.probs is not probs_to_filter : state.probs = probs_to_filter

        threshold_val = None
        if state.state == SS.PROBS_N_S:
            if state.indices is None: logger.error("SS_MinP: state.indices is None for PROBS_N_S"); return
            threshold_val = probs_to_filter[:, :1] * self.min_p
        else:
            threshold_val = probs_to_filter.amax(dim = -1, keepdim = True) * self.min_p

        mask = probs_to_filter >= threshold_val
        probs_to_filter *= mask

        state.state = SS.PROBS_S if state.state == SS.PROBS_N_S else SS.PROBS

    def prep(self, in_state: SS) -> Optional[List[type['SS_Base']]]:
        if in_state not in [SS.PROBS_N, SS.PROBS_N_S]:
            return [SS_Normalize]
        return None

    def alt(self) -> Optional[SS_Base]:
        if abs(self.min_p) < 1e-9:
            return SS_NoOp()
        return None