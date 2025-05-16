import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from .sample import SS_Sample # Inherits from SS_Sample, now in a sibling module
# SS_Normalize is used in prep(). Assume it's still in ss_definitions for now.
from ..ss_definitions import SamplingState, SS, SS_Base
from .normalize import SS_Normalize # Import SS_Normalize from its new sibling module

logger = logging.getLogger(__name__)

class SS_Sample_mn(SS_Sample):
    def run(self, state: SamplingState):
        if state.sample is not None:
            state.state = SS.DONE
            return

        if state.state not in [SS.PROBS_N_S, SS.PROBS_N]:
            error_msg = f"SS_Sample_mn: Invalid input state {state.state}. Requires PROBS_N or PROBS_N_S."
            logger.error(error_msg)
            current_probs_for_multinomial = None
            if state.state in [SS.INIT, SS.LOGITS, SS.LOGITS_S] and (state.logits or state.in_logits):
                source = state.logits if state.logits is not None else state.in_logits
                if source is not None:
                    logger.warning(f"SS_Sample_mn: Input is logits ({state.state}). Normalizing for multinomial sampling.")
                    current_probs_for_multinomial = torch.softmax(source.float(), dim=-1)
            elif state.state in [SS.PROBS, SS.PROBS_S] and state.probs is not None:
                logger.warning(f"SS_Sample_mn: Input is unnormalized probs ({state.state}). Normalizing for multinomial sampling.")
                temp_probs_norm = state.probs.float()
                norm_factor = temp_probs_norm.sum(dim = -1, keepdim = True)
                current_probs_for_multinomial = torch.nan_to_num(temp_probs_norm / torch.where(norm_factor == 0, torch.ones_like(norm_factor), norm_factor), nan=0.0)
            
            if current_probs_for_multinomial is None:
                raise ValueError(error_msg + " Auto-conversion failed.")
            state.probs = current_probs_for_multinomial 
            state.state = SS.PROBS_N 
            state.indices = None 

        if state.probs is None: 
            logger.error(f"SS_Sample_mn: state.probs is None even after potential normalization for state {state.state}"); return

        if torch.isnan(state.probs).any() or torch.any(state.probs < 0):
            logger.error("SS_Sample_mn: state.probs contains NaN or negative values before multinomial sampling. Attempting to clean.")
            cleaned_probs = torch.nan_to_num(state.probs, nan=0.0)
            cleaned_probs = torch.clamp(cleaned_probs, min=0.0)
            norm_factor = cleaned_probs.sum(dim=-1, keepdim=True)
            final_probs_for_sampling = torch.nan_to_num(cleaned_probs / torch.where(norm_factor == 0, torch.ones_like(norm_factor), norm_factor), nan=0.0)
            if torch.all(final_probs_for_sampling == 0):
                logger.warning("SS_Sample_mn: All probabilities became zero after cleaning. Falling back to argmax of original problematic probs.")
                argmax_input = state.probs if state.indices is None else state.probs 
                temp_argmax_indices = torch.argmax(torch.nan_to_num(argmax_input,nan=-torch.inf), dim=-1, keepdim=True)
                if state.state == SS.PROBS_N_S and state.indices is not None : 
                     state.sample = state.indices.gather(dim=-1, index=temp_argmax_indices).squeeze(-1)
                else: 
                     state.sample = temp_argmax_indices.squeeze(-1)
            else:
                try:
                    state.sample = torch.multinomial(final_probs_for_sampling, num_samples=1)
                except RuntimeError as e_multi: 
                    logger.error(f"SS_Sample_mn: torch.multinomial failed after cleaning: {e_multi}. Falling back to argmax.")
                    argmax_input = final_probs_for_sampling
                    temp_argmax_indices = torch.argmax(argmax_input, dim=-1, keepdim=True) 
                    if state.state == SS.PROBS_N_S and state.indices is not None :
                        state.sample = state.indices.gather(dim=-1, index=temp_argmax_indices).squeeze(-1)
                    else:
                        state.sample = temp_argmax_indices.squeeze(-1)

        else: 
            try:
                state.sample = torch.multinomial(state.probs, num_samples=1)
            except RuntimeError as e:
                logger.error(f"SS_Sample_mn: torch.multinomial failed: {e}. This can happen if all probabilities in a row are zero.")
                logger.warning("SS_Sample_mn: Falling back to argmax due to multinomial error (all probs zero?).")
                argmax_input = state.probs
                temp_argmax_indices = torch.argmax(argmax_input, dim=-1, keepdim=True)
                if state.state == SS.PROBS_N_S and state.indices is not None :
                     state.sample = state.indices.gather(dim=-1, index=temp_argmax_indices).squeeze(-1)
                else:
                     state.sample = temp_argmax_indices.squeeze(-1)
        
        state.state = SS.DONE

    def prep(self, in_state: SS) -> Optional[List[type['SS_Base']]]:
        if in_state not in [SS.PROBS_N, SS.PROBS_N_S]:
            if in_state in [SS.INIT, SS.LOGITS, SS.LOGITS_S, SS.PROBS, SS.PROBS_S]:
                return [SS_Normalize] 
        return None