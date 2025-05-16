import torch
from typing import Optional, List, Set, TYPE_CHECKING, Deque
from collections import deque
from itertools import islice
import logging

# Assuming SS_Base, SamplingState, SS will be in ss_definitions.py or a base module
# For now, let's assume they will be accessible from a level up if ss_definitions.py becomes the aggregator
from ..ss_definitions import SS_Base, SamplingState, SS # Placeholder, might change to ..base or similar
from ..utils.token_utils import NgramNode, get_dry_default_sequence_breaker_tokens

if TYPE_CHECKING:
    from ..custom import CustomSampler # Corrected path

logger = logging.getLogger(__name__)

class SS_DRY(SS_Base):
    def __init__(
        self,
        multiplier: float,
        base: float,
        allowed_length: int,
        sequence_breakers: Optional[Set[int]] = None,
        range_val: int = 0,
        max_ngram: int = 10
    ):
        super().__init__()
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.user_sequence_breakers = sequence_breakers
        self.range_val = range_val
        self.max_ngram = max(2, max_ngram)

        self.sampler: Optional['CustomSampler'] = None # Forward reference
        self.key: str = f"ss_dry_state_{id(self)}"
        self._resolved_sequence_breakers: Optional[Set[int]] = None

    def attach_sampler(self, sampler: 'CustomSampler'): # Forward reference
        self.sampler = sampler
        if self.key not in self.sampler.states:
            self.sampler.states[self.key] = []

    def _ensure_batch_state(self, bsz: int):
        if self.sampler is None:
            raise RuntimeError("SS_DRY.sampler is not attached.")
        
        current_batch_states = self.sampler.states.get(self.key)
        if current_batch_states is None:
             self.sampler.states[self.key] = []
             current_batch_states = self.sampler.states[self.key]

        if len(current_batch_states) < bsz:
            for _ in range(bsz - len(current_batch_states)):
                current_batch_states.append({
                    "ngram_trie": NgramNode(value=0),
                    "ngram_history": deque(),
                    "processed_len": 0
                })
        elif len(current_batch_states) > bsz:
            self.sampler.states[self.key] = current_batch_states[:bsz]

    def _get_resolved_sequence_breakers(self) -> Set[int]:
        if self.sampler is None or self.sampler.tokenizer is None:
            return set()
        if self._resolved_sequence_breakers is None:
            if self.user_sequence_breakers is not None:
                self._resolved_sequence_breakers = self.user_sequence_breakers
            else:
                self._resolved_sequence_breakers = get_dry_default_sequence_breaker_tokens(self.sampler.tokenizer)
        return self._resolved_sequence_breakers

    def run(self, state: SamplingState):
        #logger.debug(f"SS_DRY.run: Called. Multiplier: {self.multiplier}")
        if self.sampler is None:
            raise RuntimeError("SS_DRY.sampler is not attached. Call attach_sampler first.")
        if self.multiplier > 0.0 and self.sampler.tokenizer is None:
            raise RuntimeError("SS_DRY is active (multiplier > 0) but sampler.tokenizer is None. Tokenizer is required for sequence breakers.")

        if self.multiplier <= 0.0:
            if state.state == SS.INIT:
                if state.in_logits is None:
                    logger.error("SS_DRY.run: state.in_logits is None in SS.INIT state even when DRY is inactive.")
                    return
                if state.logits is None: state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return

        if state.state == SS.INIT:
            if state.in_logits is None:
                logger.error("SS_DRY.run: state.in_logits is None in SS.INIT state.")
                return
            state.logits = state.in_logits.clone().float()
        elif state.logits is None:
            logger.error("SS_DRY.run: state.logits is None and state is not SS.INIT. SS_DRY requires logits.")
            raise ValueError("SS_DRY requires input logits. Ensure it's placed correctly in the sampler chain.")
        elif state.logits is state.in_logits and state.in_logits is not None: # Ensure we're not modifying in_logits directly if it was passed as logits
            state.logits = state.in_logits.clone().float()


        self._ensure_batch_state(state.bsz)
        all_batch_dry_states = self.sampler.states[self.key]
        sequence_breakers = self._get_resolved_sequence_breakers()

        #logger.debug(f"SS_DRY.run: Processing batch. bsz: {state.bsz}, past_ids shape: {state.past_ids.shape if state.past_ids is not None else 'None'}")
        if logger.isEnabledFor(logging.DEBUG): # Avoid formatting if not debugging
            logger.debug(f"SS_DRY.run: state.dim: {state.dim}")
            logger.debug(f"SS_DRY.run: state.logits.shape before loop: {state.logits.shape if state.logits is not None else 'None'}")
        for i in range(state.bsz):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"SS_DRY.run: Loop i={i}, state.logits.shape at loop start: {state.logits.shape if state.logits is not None else 'None'}")
            current_logits_item = state.logits[i] # type: ignore
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"SS_DRY.run: Loop i={i}, current_logits_item.shape: {current_logits_item.shape if current_logits_item is not None else 'None'}")
            if state.past_ids is None or state.past_ids.shape[1] == 0:
                continue

            current_sequence_tensor = state.past_ids[i]
            current_sequence_list = current_sequence_tensor.tolist()
            
            if len(current_sequence_list) < self.allowed_length:
                continue

            batch_dry_state = all_batch_dry_states[i]
            ngram_hist_deque = batch_dry_state["ngram_history"] # type: ignore

            if "processed_len" not in batch_dry_state: batch_dry_state["processed_len"] = 0
            
            processed_len = batch_dry_state["processed_len"] # type: ignore
            if len(current_sequence_list) < processed_len:
                #logger.debug(f"SS_DRY.run: Batch item {i}. Context reset detected. Old processed_len: {processed_len}, new seq_len: {len(current_sequence_list)}")
                batch_dry_state["ngram_trie"] = NgramNode(value=0)
                ngram_hist_deque.clear()
                batch_dry_state["processed_len"] = 0
                processed_len = 0

            new_tokens_to_process = current_sequence_list[processed_len:]

            if new_tokens_to_process:
                #logger.debug(f"SS_DRY.run: Batch item {i}. Processing {len(new_tokens_to_process)} new tokens for trie update.")
                for token_val in new_tokens_to_process:
                    ngram_hist_deque.append(token_val)
                    if self.range_val > 0 and len(ngram_hist_deque) > self.range_val:
                        ngram_hist_deque.popleft()
                    
                    len_deque = len(ngram_hist_deque)
                    start_idx_trie = max(0, len_deque - self.max_ngram)
                    current_hist_for_trie_window = list(islice(ngram_hist_deque, start_idx_trie, len_deque))

                    for length in range(2, self.max_ngram + 1):
                        if len(current_hist_for_trie_window) >= length:
                            ngram = current_hist_for_trie_window[-length:]
                            node = batch_dry_state["ngram_trie"] # type: ignore
                            valid_ngram_path = True
                            for t_in_ngram in ngram:
                                if t_in_ngram in sequence_breakers:
                                    valid_ngram_path = False
                                    break
                                if t_in_ngram not in node.children:
                                    node.children[t_in_ngram] = NgramNode(value=0)
                                node = node.children[t_in_ngram]
                            
                            if valid_ngram_path:
                                node.value += 1
                                #logger.debug(f"SS_DRY.run: Batch item {i}. Trie updated. Ngram: {ngram}, New count: {node.value}")
                
                batch_dry_state["processed_len"] = len(current_sequence_list)

            penalized_in_this_step = set()
            
            len_deque_penalty = len(ngram_hist_deque)
            hist_len_for_penalty_prefix = self.max_ngram -1 
            start_idx_penalty = max(0, len_deque_penalty - hist_len_for_penalty_prefix)
            history_for_penalty_window = list(islice(ngram_hist_deque, start_idx_penalty, len_deque_penalty))
            
            #logger.debug(f"SS_DRY.run: Batch item {i}. Applying penalties. Effective history window length for penalty prefix: {len(history_for_penalty_window)}")

            for ngram_len_to_check in range(2, self.max_ngram + 1):
                prefix_len_needed = ngram_len_to_check - 1
                if len(history_for_penalty_window) < prefix_len_needed:
                    continue

                history_prefix = history_for_penalty_window[-prefix_len_needed:]
                
                node_after_prefix = batch_dry_state["ngram_trie"] # type: ignore
                path_exists_in_trie = True
                for token_in_prefix in history_prefix:
                    if token_in_prefix in node_after_prefix.children:
                        node_after_prefix = node_after_prefix.children[token_in_prefix]
                    else:
                        path_exists_in_trie = False
                        break
                
                if path_exists_in_trie:
                    for token_id_completing_ngram, child_node in node_after_prefix.children.items():
                        if token_id_completing_ngram not in penalized_in_this_step:
                            count = child_node.value
                            if count > 0:
                                exc_length = ngram_len_to_check - self.allowed_length
                                current_base = self.base if self.base > 1e-6 else 1.0
                                penalty_scaling_factor = current_base ** max(0, exc_length)
                                penalty_amount = self.multiplier * penalty_scaling_factor * count
                                
                                if penalty_amount > 0:
                                    #original_logit_val = current_logits_item[0, token_id_completing_ngram].item()
                                    current_logits_item[token_id_completing_ngram] -= penalty_amount
                                    penalized_in_this_step.add(token_id_completing_ngram)
                                    #logger.debug(f"SS_DRY.run: Batch item {i}. Penalized token {token_id_completing_ngram} (ngram_len: {ngram_len_to_check}, exc_len: {exc_length}, count: {count}, scale_factor: {penalty_scaling_factor:.2f}, penalty: {penalty_amount:.2f}). Logit: {original_logit_val:.4f} -> {current_logits_item[token_id_completing_ngram]:.4f}")

        #logger.debug(f"SS_DRY.run: Finished processing. Final state for logits: {state.state}")
        state.state = SS.LOGITS

    def reqs_past_ids(self):
        return True