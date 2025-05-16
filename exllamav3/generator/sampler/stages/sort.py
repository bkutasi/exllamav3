import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from ..ss_definitions import SS_Base, SamplingState, SS

logger = logging.getLogger(__name__)

class SS_Sort(SS_Base):
    def run(self, state: SamplingState):
        already_sorted_and_consistent = False
        if state.state == SS.LOGITS_S:
            if state.logits is not None and state.indices is not None: already_sorted_and_consistent = True
        elif state.state == SS.PROBS_S or state.state == SS.PROBS_N_S:
            if state.probs is not None and state.indices is not None: already_sorted_and_consistent = True
        
        if already_sorted_and_consistent:
            return

        if state.state in [SS.LOGITS_S, SS.PROBS_S, SS.PROBS_N_S] and not already_sorted_and_consistent:
            logger.warning(f"SS_Sort: State is {state.state} but fields (logits/probs or indices) are inconsistent. Re-sorting.")

        sorted_values = None
        sorted_indices = None
        output_state = None
        current_tensor_to_sort = None

        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Sort: state.in_logits is None for SS.INIT"); return
            current_tensor_to_sort = state.in_logits.float() 
            output_state = SS.LOGITS_S
            state.logits = current_tensor_to_sort 
        elif state.state == SS.LOGITS or state.state == SS.LOGITS_S: 
            if state.logits is None: 
                if state.in_logits is not None and state.state == SS.LOGITS : 
                    logger.warning(f"SS_Sort: state.logits is None for {state.state}, using state.in_logits.")
                    current_tensor_to_sort = state.in_logits.float()
                    state.logits = current_tensor_to_sort 
                else:
                    logger.error(f"SS_Sort: state.logits is None for {state.state} and cannot recover."); return
            else: 
                 current_tensor_to_sort = state.logits.float() 
                 if state.logits is not current_tensor_to_sort : state.logits = current_tensor_to_sort 
            output_state = SS.LOGITS_S
        elif state.state in [SS.PROBS, SS.PROBS_N, SS.PROBS_S, SS.PROBS_N_S]: 
            if state.probs is None:  
                 logger.error(f"SS_Sort: state.probs is None for {state.state}"); return
            current_tensor_to_sort = state.probs.float() 
            if state.probs is not current_tensor_to_sort: state.probs = current_tensor_to_sort 
            
            if state.state == SS.PROBS: output_state = SS.PROBS_S
            elif state.state == SS.PROBS_N: output_state = SS.PROBS_N_S
            elif state.state == SS.PROBS_S: output_state = SS.PROBS_S 
            elif state.state == SS.PROBS_N_S: output_state = SS.PROBS_N_S 
        else:
            logger.error(f"SS_Sort: Invalid input state {state.state} for sorting.")
            return

        sorted_values, sorted_indices = torch.sort(current_tensor_to_sort, dim = -1, descending = True)

        if output_state == SS.LOGITS_S:
            state.logits = sorted_values
        elif output_state in [SS.PROBS_S, SS.PROBS_N_S]:
            state.probs = sorted_values
        
        state.indices = sorted_indices
        state.state = output_state