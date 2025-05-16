import torch
from typing import Optional, List # For SS_Base.prep type hint
import logging

from exllamav3.ext import exllamav3_ext as ext # Crucial import for this class
from ..ss_definitions import SS_Base, SamplingState, SS

logger = logging.getLogger(__name__)

class SS_Sample(SS_Base):
    def run(self, state: SamplingState):
        if state.sample is not None:
            state.state = SS.DONE
            return

        sampled_tokens = None
        processed_tensor = None
        is_sorted_input = False

        if state.state == SS.INIT:
            if state.in_logits is None: logger.error("SS_Sample: state.in_logits is None for SS.INIT"); return
            processed_tensor = state.in_logits.float()
            ext.gumbel_noise_f32(processed_tensor, processed_tensor, state.rand_u32)
        elif state.state == SS.LOGITS:
            if state.logits is None: logger.error("SS_Sample: state.logits is None for SS.LOGITS"); return
            processed_tensor = state.logits.float()
            ext.gumbel_noise_f32(processed_tensor, processed_tensor, state.rand_u32)
        elif state.state == SS.LOGITS_S:
            if state.logits is None: logger.error("SS_Sample: state.logits is None for SS.LOGITS_S"); return
            processed_tensor = state.logits.float()
            ext.gumbel_noise_f32(processed_tensor, processed_tensor, state.rand_u32)
            is_sorted_input = True
        elif state.state in [SS.PROBS, SS.PROBS_N]:
            if state.probs is None: logger.error(f"SS_Sample: state.probs is None for {state.state}"); return
            if torch.isnan(state.probs).any():
                logger.error(f"SS_Sample: state.probs contains NaNs for state {state.state}. Attempting to use logits if available.")
                if state.logits is not None:
                    logger.warning("SS_Sample: Falling back to sampling from state.logits due to NaN in state.probs.")
                    processed_tensor = state.logits.float()
                    ext.gumbel_noise_f32(processed_tensor, processed_tensor, state.rand_u32)
                    is_sorted_input = state.state in [SS.LOGITS_S, SS.PROBS_S, SS.PROBS_N_S]
                elif state.in_logits is not None:
                    logger.warning("SS_Sample: Falling back to sampling from state.in_logits due to NaN in state.probs and no state.logits.")
                    processed_tensor = state.in_logits.float()
                    ext.gumbel_noise_f32(processed_tensor, processed_tensor, state.rand_u32)
                    is_sorted_input = False
                else:
                    logger.critical("SS_Sample: state.probs contains NaNs, and no fallback logits available. Sampling will likely fail or produce incorrect results.")
                    processed_tensor = torch.nan_to_num(state.probs.float(), nan=0.0)
                    if torch.all(processed_tensor == 0): logger.warning("SS_Sample: All probs became 0 after nan_to_num. Sampling will pick based on Gumbel noise alone.")
                    ext.gumbel_noise_log(processed_tensor, processed_tensor, state.rand_u32)
            else:
                processed_tensor = state.probs.float()
                ext.gumbel_noise_log(processed_tensor, processed_tensor, state.rand_u32)
        elif state.state in [SS.PROBS_S, SS.PROBS_N_S]:
            if state.probs is None: logger.error(f"SS_Sample: state.probs is None for {state.state}"); return
            if torch.isnan(state.probs).any():
                logger.error(f"SS_Sample: state.probs contains NaNs for sorted state {state.state}. Attempting fallback.")
                if state.logits is not None:
                     logger.warning("SS_Sample: Falling back to sampling from state.logits (sorted) due to NaN in state.probs.")
                     processed_tensor = state.logits.float()
                     ext.gumbel_noise_f32(processed_tensor, processed_tensor, state.rand_u32)
                else:
                     logger.critical(f"SS_Sample: state.probs contains NaNs for sorted state {state.state}, no sorted logits for fallback. Trying nan_to_num.")
                     processed_tensor = torch.nan_to_num(state.probs.float(), nan=0.0)
                     if torch.all(processed_tensor == 0): logger.warning("SS_Sample: All sorted probs became 0 after nan_to_num.")
                     ext.gumbel_noise_log(processed_tensor, processed_tensor, state.rand_u32)
            else:
                processed_tensor = state.probs.float()
                ext.gumbel_noise_log(processed_tensor, processed_tensor, state.rand_u32)
            is_sorted_input = True
        else:
            logger.error(f"SS_Sample: Invalid input state {state.state}")
            return

        if processed_tensor is None:
            logger.error("SS_Sample: processed_tensor is None before argmax. Cannot sample."); return

        if is_sorted_input:
            if state.indices is None: logger.error(f"SS_Sample: state.indices is None for sorted input from state {state.state}"); return
            if torch.isnan(processed_tensor).any():
                logger.error(f"SS_Sample: Processed tensor for argmax (sorted) contains NaNs. This may lead to selecting token 0. Original state: {state.state}")
                processed_tensor = torch.nan_to_num(processed_tensor, nan=-torch.inf)

            temp_argmax_indices = torch.argmax(processed_tensor, dim = -1, keepdim=True)
            sampled_tokens = state.indices.gather(dim=-1, index=temp_argmax_indices).squeeze(-1)
        else:
            if torch.isnan(processed_tensor).any():
                logger.error(f"SS_Sample: Processed tensor for argmax (unsorted) contains NaNs. This may lead to selecting token 0. Original state: {state.state}")
                processed_tensor = torch.nan_to_num(processed_tensor, nan=-torch.inf)
            sampled_tokens = torch.argmax(processed_tensor, dim = -1)

        state.sample = sampled_tokens
        state.state = SS.DONE