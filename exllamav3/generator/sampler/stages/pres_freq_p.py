import torch # For SamplingState potentially
from typing import Optional, List # For SS_Base.prep type hint
import logging

from exllamav3.ext import exllamav3_ext as ext # For apply_pres_freq_pens
from ..ss_definitions import SS_Base, SamplingState, SS
from .noop import SS_NoOp # For alt() method

logger = logging.getLogger(__name__)

class SS_PresFreqP(SS_Base):
    def __init__(
        self,
        pres_p: float = 0.0,
        freq_p: float = 0.0,
        sustain_range: int = int(10e7),
        decay_range: int = 0
    ):
        self.pres_p = pres_p
        self.freq_p = freq_p
        self.sustain_range = max(0, sustain_range)
        self.decay_range = max(0, decay_range)

    def run(self, state: SamplingState):
        if abs(self.pres_p) < 1e-9 and abs(self.freq_p) < 1e-9 or \
           (self.sustain_range == 0 and self.decay_range == 0):
            if state.state == SS.INIT and state.in_logits is not None and state.logits is None:
                state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return

        if state.past_ids is None and (self.sustain_range > 0 or self.decay_range > 0):
            logger.warning("SS_PresFreqP: past_ids is None. Penalties will not be applied effectively.")
            if state.state == SS.INIT and state.in_logits is not None and state.logits is None:
                state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return

        input_logits_for_kernel = None
        original_state_was_sorted_logits = False
        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_PresFreqP: state.in_logits is None for SS.INIT"); return
            input_logits_for_kernel = state.in_logits
        elif state.state == SS.LOGITS:
            if state.logits is None: logger.error("SS_PresFreqP: state.logits is None for SS.LOGITS"); return
            input_logits_for_kernel = state.logits
        elif state.state == SS.LOGITS_S:
            if state.logits is None: logger.error("SS_PresFreqP: state.logits is None for SS.LOGITS_S"); return
            input_logits_for_kernel = state.logits
            original_state_was_sorted_logits = True
        else:
            logger.error(f"SS_PresFreqP: Invalid input state {state.state}. Requires INIT, LOGITS, or LOGITS_S.")
            if state.in_logits is not None:
                logger.warning(f"SS_PresFreqP: Input state was {state.state}. Applying penalty to original in_logits.")
                input_logits_for_kernel = state.in_logits
            else:
                raise ValueError("SS_PresFreqP must operate on logits. No suitable logits found.")

        output_logits_buffer = torch.empty_like(input_logits_for_kernel, dtype=torch.float)

        ext.apply_pres_freq_pens(
            input_logits_for_kernel,
            output_logits_buffer,
            state.past_ids,
            self.pres_p,
            self.freq_p,
            self.sustain_range,
            self.decay_range
        )
        state.logits = output_logits_buffer
        state.state = SS.LOGITS
        if original_state_was_sorted_logits:
            state.indices = None


    def alt(self):
        if (abs(self.pres_p) < 1e-9 and abs(self.freq_p) < 1e-9) or \
           (self.sustain_range == 0 and self.decay_range == 0):
            return SS_NoOp()
        return None

    def reqs_past_ids(self):
        return self.sustain_range > 0 or self.decay_range > 0