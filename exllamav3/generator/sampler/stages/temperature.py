import torch # For SamplingState potentially
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS
from .noop import SS_NoOp # SS_NoOp is used in alt()

logger = logging.getLogger(__name__)

class SS_Temperature(SS_Base):
    def __init__(self, temperature: float, min_temp: float = 0.0, max_temp: float = 0.0, exponent: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.min_temp = min_temp if min_temp > 1e-9 else 0.0
        self.max_temp = max_temp if max_temp > 1e-9 else 0.0
        self.exponent = exponent
        if self.min_temp > 0 and self.max_temp > 0 and self.min_temp > self.max_temp:
            logger.warning(f"SS_Temperature: min_temp {self.min_temp} > max_temp {self.max_temp}. Clamping behavior might be unexpected.")

    def run(self, state: SamplingState):
        eff_temp = self.temperature
        if self.min_temp > 1e-9:
            eff_temp = max(eff_temp, self.min_temp)
        if self.max_temp > 1e-9:
            eff_temp = min(eff_temp, self.max_temp)

        if eff_temp < 1e-9: eff_temp = 1e-9

        eff_temp_adj = eff_temp ** self.exponent
        if eff_temp_adj < 1e-9: eff_temp_adj = 1e-9


        is_no_op_temp = abs(eff_temp_adj - 1.0) < 1e-9
        if is_no_op_temp:
            if state.state == SS.INIT and state.in_logits is not None and state.logits is None:
                state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return


        target_logits = None
        target_probs = None
        input_is_logits = False
        input_is_probs = False

        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Temperature: state.in_logits is None for SS.INIT"); return
            target_logits = state.in_logits.clone().float()
            input_is_logits = True
        elif state.state in [SS.LOGITS, SS.LOGITS_S]:
            if state.logits is None:
                logger.error(f"SS_Temperature: state.logits is None in {state.state} state."); return
            if state.logits is state.in_logits and state.in_logits is not None:
                target_logits = state.in_logits.clone().float()
            else:
                target_logits = state.logits.float()
            input_is_logits = True
        elif state.state in [SS.PROBS, SS.PROBS_N, SS.PROBS_S, SS.PROBS_N_S]:
            if state.probs is None: logger.error(f"SS_Temperature: state.probs is None for {state.state} state."); return
            target_probs = state.probs.clone().float()
            input_is_probs = True
        else:
            logger.error(f"SS_Temperature: Unexpected state {state.state}")
            if state.in_logits is not None:
                target_logits = state.in_logits.clone().float()
                input_is_logits = True
            else: return

        if input_is_logits and target_logits is not None:
            target_logits /= eff_temp_adj
            state.logits = target_logits
            state.state = SS.LOGITS_S if state.state == SS.LOGITS_S else SS.LOGITS
        elif input_is_probs and target_probs is not None:
            target_probs.pow_(1.0 / eff_temp_adj)
            state.probs = target_probs
            state.state = SS.PROBS_S if state.state in [SS.PROBS_S, SS.PROBS_N_S] else SS.PROBS
        else:
            logger.error("SS_Temperature: Logic error, no target tensor to operate on."); return


    def alt(self):
        eff_temp = self.temperature
        if self.min_temp > 1e-9: eff_temp = max(eff_temp, self.min_temp)
        if self.max_temp > 1e-9: eff_temp = min(eff_temp, self.max_temp)
        if eff_temp < 1e-9: eff_temp = 1e-9
        eff_temp_adj = eff_temp ** self.exponent
        if eff_temp_adj < 1e-9: eff_temp_adj = 1e-9

        if abs(eff_temp_adj - 1.0) < 1e-9 :
            return SS_NoOp()
        return None