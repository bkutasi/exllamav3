import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS

logger = logging.getLogger(__name__)

class SS_Normalize(SS_Base):
    def run(self, state: SamplingState):
        if state.state == SS.PROBS_N or state.state == SS.PROBS_N_S:
            if state.probs is None:
                 logger.error(f"SS_Normalize: state is {state.state} but state.probs is None. This indicates a logic error.")
                 if state.logits is not None: 
                    logger.warning("SS_Normalize: Recovering state.probs from state.logits.")
                    state.probs = torch.softmax(state.logits.float(), dim=-1)
                 elif state.in_logits is not None and state.state == SS.PROBS_N: 
                    logger.warning("SS_Normalize: Recovering state.probs from state.in_logits for PROBS_N.")
                    state.probs = torch.softmax(state.in_logits.float(), dim=-1)
                 else: 
                    logger.error(f"SS_Normalize: Cannot recover state.probs for {state.state}.")
                    return 
            return 

        target_probs = None
        output_state = None
        source_for_softmax = None 
        source_for_renorm = None  

        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Normalize: state.in_logits is None for SS.INIT"); return
            source_for_softmax = state.in_logits
            output_state = SS.PROBS_N
            if state.logits is None: state.logits = state.in_logits.float() 
        elif state.state == SS.LOGITS:
            if state.logits is None: logger.error("SS_Normalize: state.logits is None for SS.LOGITS"); return
            source_for_softmax = state.logits
            output_state = SS.PROBS_N
        elif state.state == SS.PROBS: 
            if state.probs is None: logger.error("SS_Normalize: state.probs is None for SS.PROBS"); return
            source_for_renorm = state.probs
            output_state = SS.PROBS_N
        elif state.state == SS.LOGITS_S:
            if state.logits is None or state.indices is None: logger.error("SS_Normalize: state.logits or state.indices is None for SS.LOGITS_S"); return
            source_for_softmax = state.logits
            output_state = SS.PROBS_N_S
        elif state.state == SS.PROBS_S: 
            if state.probs is None or state.indices is None: logger.error("SS_Normalize: state.probs or state.indices is None for SS.PROBS_S"); return
            source_for_renorm = state.probs
            output_state = SS.PROBS_N_S
        else: 
            logger.error(f"SS_Normalize: Invalid input state {state.state}")
            return

        if source_for_softmax is not None:
            target_probs = torch.softmax(source_for_softmax.float(), dim = -1)
        elif source_for_renorm is not None:
            probs_to_renorm = source_for_renorm.float() 
            norm_factor = probs_to_renorm.sum(dim = -1, keepdim = True)
            target_probs = torch.nan_to_num(probs_to_renorm / torch.where(norm_factor == 0, torch.ones_like(norm_factor), norm_factor), nan=0.0)
        else: 
            logger.error("SS_Normalize: No source tensor determined for normalization."); return
            
        state.probs = target_probs
        state.state = output_state