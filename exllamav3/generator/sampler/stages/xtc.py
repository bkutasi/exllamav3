import torch
import random
from typing import Optional, Set, TYPE_CHECKING, List # Added List for SS_Base prep method
import logging

from ..ss_definitions import SS_Base, SamplingState, SS
from ..utils.token_utils import get_xtc_default_ignore_tokens

if TYPE_CHECKING:
    from exllamav3.generator.sampler.custom import CustomSampler # For type hinting

logger = logging.getLogger(__name__)

class SS_XTC(SS_Base):
    def __init__(
        self,
        probability: float,
        threshold: float,
        ignore_tokens: Optional[Set[int]] = None,
    ):
        super().__init__()
        self.probability = probability
        self.threshold = threshold
        self.user_ignore_tokens = ignore_tokens

        self.sampler_ref: Optional['CustomSampler'] = None # Forward reference
        self._cached_xtc_ignore_mask: Optional[torch.Tensor] = None
        self._cached_tokenizer_id: Optional[int] = None
        self._cached_vocab_size: Optional[int] = None
        self._cached_device: Optional[torch.device] = None

    def attach_sampler(self, sampler: 'CustomSampler'): # Forward reference
        logger.debug(f"SS_XTC.attach_sampler: Called with sampler {id(sampler)}")
        self.sampler_ref = sampler

    def _get_resolved_ignore_tokens(self) -> Set[int]:
        logger.debug("SS_XTC._get_resolved_ignore_tokens: Entered.")
        if self.user_ignore_tokens is not None:
            logger.debug(f"SS_XTC._get_resolved_ignore_tokens: Using {len(self.user_ignore_tokens)} user_ignore_tokens.")
            return self.user_ignore_tokens
        if self.sampler_ref and self.sampler_ref.tokenizer:
            logger.debug("SS_XTC._get_resolved_ignore_tokens: Calling get_xtc_default_ignore_tokens.")
            default_tokens = get_xtc_default_ignore_tokens(self.sampler_ref.tokenizer)
            logger.debug(f"SS_XTC._get_resolved_ignore_tokens: Got {len(default_tokens)} default tokens from utility.")
            return default_tokens
        logger.warning("SS_XTC: Tokenizer not available for default ignore tokens. Using empty set.")
        return set()

    def _ensure_xtc_ignore_mask(self, device: torch.device, vocab_size: int):
        logger.debug(f"SS_XTC._ensure_xtc_ignore_mask: Called. Device: {device}, Vocab size: {vocab_size}")
        logger.debug(f"SS_XTC._ensure_xtc_ignore_mask: self.sampler_ref is {'set' if self.sampler_ref else 'None'}")
        if self.sampler_ref:
            logger.debug(f"SS_XTC._ensure_xtc_ignore_mask: self.sampler_ref.tokenizer is {'set' if self.sampler_ref.tokenizer else 'None'}")
        tokenizer_available = self.sampler_ref and self.sampler_ref.tokenizer
        current_tokenizer_id = id(self.sampler_ref.tokenizer) if tokenizer_available else -1

        if (self._cached_xtc_ignore_mask is None or
            self._cached_tokenizer_id != current_tokenizer_id or
            self._cached_vocab_size != vocab_size or
            self._cached_device != device):

            if not tokenizer_available:
                logger.warning("SS_XTC: Tokenizer is required to build ignore mask but not available. XTC will effectively allow all tokens to be filtered (potentially undesirable).")
                self._cached_xtc_ignore_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
            else:
                ignore_token_ids = self._get_resolved_ignore_tokens()
                logger.debug(f"SS_XTC._ensure_xtc_ignore_mask: Resolved {len(ignore_token_ids)} ignore_token_ids.")
                mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
                if ignore_token_ids: # Only try to use if not empty
                    valid_ignore_ids = [tid for tid in ignore_token_ids if 0 <= tid < vocab_size]
                    if len(valid_ignore_ids) > 0:
                        logger.debug(f"SS_XTC._ensure_xtc_ignore_mask: Preparing to apply {len(valid_ignore_ids)} valid_ignore_ids to mask.")
                        mask[valid_ignore_ids] = False
                        logger.debug("SS_XTC._ensure_xtc_ignore_mask: Applied valid_ignore_ids to mask.")
                self._cached_xtc_ignore_mask = mask

            self._cached_tokenizer_id = current_tokenizer_id
            self._cached_vocab_size = vocab_size
            self._cached_device = device
            logger.debug(f"SS_XTC._ensure_xtc_ignore_mask: Rebuilt xtc_ignore_mask. Device: {device}, Vocab size: {vocab_size}. Tokens *not* ignored by XTC mask (candidates for filtering): {self._cached_xtc_ignore_mask.sum().item() if self._cached_xtc_ignore_mask is not None else 'N/A'}.")
            logger.debug("SS_XTC._ensure_xtc_ignore_mask: Finished.")
    def run(self, state: SamplingState):
        logger.debug(f"SS_XTC.run: Entered. Probability: {self.probability}, Threshold: {self.threshold}, Random roll: {random.random()}") # Log random roll
        current_random_roll = random.random() # Get the roll once
        logger.debug(f"SS_XTC.run: Probability: {self.probability}, Random roll for this step: {current_random_roll}")
        if self.probability == 0.0 or current_random_roll >= self.probability:
            logger.debug("SS_XTC.run: Skipped due to probability condition.")
            # logger.debug("SS_XTC: Skipped due to probability.") # Removed for less noise
            if state.state == SS.INIT:
                if state.in_logits is None:
                    logger.error("SS_XTC.run: state.in_logits is None in SS.INIT state when skipped.")
                    return
                if state.logits is None: # Only populate if not already set
                    state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return
        logger.debug("SS_XTC.run: XTC triggered.") # Log when XTC is active

        # logger.debug(f"SS_XTC: Triggered. Probability: {self.probability}, Threshold: {self.threshold}") # Removed for less noise

        if not self.sampler_ref or not self.sampler_ref.tokenizer:
            logger.error("SS_XTC: Sampler or tokenizer reference not available. Cannot apply XTC. Passing through logits.")
            if state.state == SS.INIT and state.in_logits is not None:
                if state.logits is None: state.logits = state.in_logits.float()
                state.state = SS.LOGITS
            return

        current_logits = None
        if state.state == SS.INIT:
            if state.in_logits is None:
                logger.error("SS_XTC.run: state.in_logits is None in SS.INIT state when XTC triggered.")
                return
            current_logits = state.in_logits.float() # Make a float copy
        elif state.logits is not None:
            # If state.logits is from state.in_logits, clone to make it mutable if not already cloned by a prev step
            if state.logits is state.in_logits and state.in_logits is not None:
                 current_logits = state.in_logits.clone().float()
            else: # state.logits is already a mutable tensor from a previous step
                 current_logits = state.logits.float() # Ensure float, may copy
        else:
            logger.error(f"SS_XTC.run: Logits not available for XTC. Current state: {state.state}")
            return

        bsz, vocab_size = current_logits.shape
        self._ensure_xtc_ignore_mask(device=current_logits.device, vocab_size=vocab_size)

        if self._cached_xtc_ignore_mask is None:
            logger.error("SS_XTC: Failed to create xtc_ignore_mask. Passing through logits.")
            state.logits = current_logits # Pass current_logits (which is a copy or already mutable)
            state.state = SS.LOGITS
            return

        logger.debug(f"SS_XTC: Before softmax. current_logits min: {current_logits.min().item():.4f}, max: {current_logits.max().item():.4f}, mean: {current_logits.mean().item():.4f}")
        softmax_probs = torch.softmax(current_logits, dim=-1)

        if self._cached_xtc_ignore_mask is not None:
            logger.debug(f"SS_XTC: Num XTC candidate tokens (ignore_mask sum): {self._cached_xtc_ignore_mask.sum().item()}")

        # xtc_selection_filter identifies tokens that *are* candidates for XTC (mask is True) AND pass the probability threshold.
        xtc_selection_filter = (softmax_probs > self.threshold) & self._cached_xtc_ignore_mask.unsqueeze(0) # unsqueeze for broadcasting with (bsz, vocab_size)
        logger.debug(f"SS_XTC: Num tokens passing XTC filter (xtc_selection_filter sum): {xtc_selection_filter.sum().item()}")

        # Initialize new_logits.
        # For tokens that are *not* candidates for XTC (mask is False), their original logits are preserved.
        # For tokens that *are* candidates (mask is True), they start as -inf and will be updated if they pass the threshold.
        new_logits = torch.where(
            self._cached_xtc_ignore_mask.unsqueeze(0),  # Condition: True if token is an XTC candidate
            torch.full_like(current_logits, -torch.inf), # Value if True: initially filter out
            current_logits                               # Value if False: preserve original logit
        )

        # For the XTC candidate tokens that passed the threshold, restore their original logits.
        new_logits[xtc_selection_filter] = current_logits[xtc_selection_filter]

        # Fallback: Per-batch item, if all logits in a row are now -torch.inf, revert that row to original logits.
        for i in range(new_logits.shape[0]): # Iterate over batch items
            if torch.all(new_logits[i] == -torch.inf):
                logger.warning(f"SS_XTC: Batch item {i}: All tokens were filtered to -inf by XTC (threshold {self.threshold}). Reverting to original logits for this item.")
                logger.debug(f"SS_XTC: Fallback for item {i}. current_logits[i] min: {current_logits[i].min().item():.4f}, max: {current_logits[i].max().item():.4f}, mean: {current_logits[i].mean().item():.4f}")
                new_logits[i] = current_logits[i] # current_logits is the input to XTC for this step

        state.logits = new_logits
        logger.debug(f"SS_XTC: After XTC. state.logits min: {state.logits.min().item():.4f}, max: {state.logits.max().item():.4f}, mean: {state.logits.mean().item():.4f}")
        state.state = SS.LOGITS
        # logger.debug(f"SS_XTC: Applied. Tokens passing XTC filter criteria: {xtc_selection_filter.sum().item()}. " # Removed for less noise
        #              f"Logits shape: {state.logits.shape}")

    def reqs_past_ids(self):
        return False