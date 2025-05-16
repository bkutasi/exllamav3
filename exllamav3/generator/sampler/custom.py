from .sampler import Sampler
import torch
from typing_extensions import override
from ...tokenizer import Tokenizer
import random
from .ss_definitions import SS_Base, SamplingState, SS
from .stages import SS_Sample # Added import for DefaultSampler

class CustomSampler(Sampler):
    def __init__(
        self,
        steps: list[SS_Base]
    ):
        super().__init__()
        self.states = {} # Initialize the states dictionary
        self.tokenizer: Tokenizer | None = None # Initialize tokenizer attribute

        self.steps = []
        # Initialize reqs_past_ids, it's a boolean attribute of the Sampler instance
        self.reqs_past_ids = False # Initialize here
        current_processing_state = SS.INIT # Use a distinct variable for the loop's state tracking
        for step_item in steps: # Use a different variable name for the loop item
            self.reqs_past_ids = self.reqs_past_ids or step_item.reqs_past_ids()
            alt_step = step_item.alt() # Use a different variable name
            if alt_step:
                current_step_to_process = alt_step # Use a different variable name
            else:
                current_step_to_process = step_item

            prep_steps_classes = current_step_to_process.prep(current_processing_state) # Use a different variable name
            if prep_steps_classes:
                for prep_step_class in prep_steps_classes:
                    new_prep_step_instance = prep_step_class()
                    if hasattr(new_prep_step_instance, 'attach_sampler') and callable(getattr(new_prep_step_instance, 'attach_sampler')):
                        new_prep_step_instance.attach_sampler(self)
                    self.steps.append(new_prep_step_instance)
            
            if hasattr(current_step_to_process, 'attach_sampler') and callable(getattr(current_step_to_process, 'attach_sampler')):
                current_step_to_process.attach_sampler(self)
            self.steps.append(current_step_to_process)
            # The `current_processing_state` for the *next* iteration's `step_item.prep()` call should reflect the state
            # *after* `current_step_to_process` would theoretically run. This is complex.
            # The original code used a single `state = SS.INIT` and passed it to `step.prep(state)`.
            # This `state` variable itself was not updated in the loop.
            # Let's stick to the original logic: the `state` passed to `prep` is the state *before* the current main `step_item`.
            # The `CustomSampler` doesn't know the final state after each step during __init__.
            # It just prepares the list of steps. The actual state transitions happen during `forward()`.
            # So, `current_processing_state` should not be updated here based on the step's potential outcome.
            # It should represent the state *before* the *current* `step_item` is considered for `prep`.
            # The original code had `state = SS.INIT` outside the loop and used that `state` for all `step.prep(state)` calls.
            # This is likely an oversight in the original `custom.py` or implies `prep` only cares about the initial state.
            # Let's replicate that for now, passing `SS.INIT` to `prep`.
            # prep_steps_classes = current_step_to_process.prep(SS.INIT) # Reverting to simpler logic if the above is too complex

        # Re-evaluating the state logic for prep:
        # The original code:
        # self.steps = []
        # state = SS.INIT
        # for step in steps:
        #     ...
        #     prep_steps = step.prep(state) <-- 'state' here is always SS.INIT
        #     ...
        # This implies that the `prep` method of each SS_Base class must be designed to work given only SS.INIT
        # as the input state, or the logic in `CustomSampler` was simplified.
        # Given the `prep` methods in the original `SS_TopK`, `SS_TopP` etc., they *do* match on `in_state: SS`.
        # Example: `SS_TopK.prep` matches `in_state` against `SS.INIT | SS.LOGITS | SS.PROBS | SS.PROBS_N`.
        # This means `state` in `CustomSampler.__init__` *should* be updated.
        # However, `CustomSampler.__init__` cannot know the *actual* output state of a step without running it.
        # This is a conceptual issue in the original design if `prep` truly needs the *output* state of the previous step.
        # Let's assume `prep(in_state)` refers to the state *before* the step itself runs.
        # The `state` variable in the loop should represent the expected state *before* the current `step_item`'s `prep` is called.
        # This state is not changed by `prep_steps` themselves during `__init__`.
        # The most straightforward interpretation of the original code is that `state` was indeed fixed at `SS.INIT` for all `prep` calls.
        # Let's stick to that for minimal change from original `custom.py`'s `CustomSampler.__init__` behavior.
        # The `state` variable in the loop was not updated.

    @override
    @torch.inference_mode
    def forward(
        self,
        logits, # This is the raw output from the model, effectively in_logits for the first step
        sequence_ids: torch.Tensor | None = None,
        rand_u32: int | None = None,
        tokenizer: Tokenizer | None = None,
        blocked_tokens: list[int] | None = None,
        allowed_tokens: list[int] | None = None,
        return_state: bool = False
    ):
        out_shape = logits.shape[:-1]

        # Filter out tokens outside the actual vocab size if tokenizer is provided
        if tokenizer is not None and tokenizer.actual_vocab_size < logits.shape[-1]:
            logits[..., tokenizer.actual_vocab_size:] = -float("inf")
        self.tokenizer = tokenizer # Store the tokenizer instance

        if rand_u32 is None:
            rand_u32 = random.randint(0, (1 << 32) - 1)
        # Seeding here affects torch.multinomial if used by a sampler step, and random.sample if used.
        # Note: The original exllamav2 generator seeds globally. Here, it's per-call if rand_u32 is provided.
        torch.manual_seed(rand_u32) # Ensure reproducibility if rand_u32 is given
        random.seed(rand_u32)      # Also seed Python's random for any steps relying on it

        dim = logits.shape[-1]
        bsz = logits.numel() // dim

        # Apply token blocking/allowing directly to the initial logits
        # This needs to be done carefully if logits tensor is reused.
        # Cloning here ensures modifications don't affect upstream if logits is a view or shared.
        current_logits = logits # Start with the input logits
        if blocked_tokens is not None or allowed_tokens is not None:
            current_logits = current_logits.clone() # Clone to modify
            if blocked_tokens is not None:
                current_logits[..., blocked_tokens] = float('-inf')
            if allowed_tokens is not None:
                # Create a mask for allowed tokens
                allow_mask = torch.zeros(dim, dtype=torch.bool, device=current_logits.device)
                valid_allowed_tokens = [t for t in allowed_tokens if 0 <= t < dim]
                if valid_allowed_tokens: # Ensure tokens are within vocab bounds
                    allow_mask[valid_allowed_tokens] = True
                current_logits[..., ~allow_mask] = float('-inf') # Apply mask

        # Initialize SamplingState
        sampling_state_obj = SamplingState( # Use a different variable name from the class
            rand_u32=rand_u32,
            dim=dim,
            bsz=bsz,
            in_logits=current_logits.view(bsz, dim), # Pass the potentially modified logits
            past_ids=sequence_ids,
            # Other fields (logits, probs, sample, indices) are None initially
        )

        for ss_step in self.steps:
            if sampling_state_obj.state == SS.DONE and not return_state: # Optimization: if done and not returning state, break
                break
            ss_step.run(sampling_state_obj)
            if not return_state and sampling_state_obj.state == SS.DONE and sampling_state_obj.sample is None:
                # This case should ideally not happen if DONE implies sample is set.
                # If a step sets DONE but not sample, and we don't return state, it's an issue.
                # However, the assertion below handles this.
                pass


        assert return_state or (sampling_state_obj.state == SS.DONE and sampling_state_obj.sample is not None), \
            f"Sampling logic error: Did not reach DONE state with a sample, or return_state is False. Final state: {sampling_state_obj.state}, Sample: {sampling_state_obj.sample is not None}"

        return sampling_state_obj if return_state else sampling_state_obj.sample.view(out_shape)
class DefaultSampler(CustomSampler):
    def __init__(self):
        # Initialize with a basic sampling step, e.g., SS_Sample
        # You might need to import SS_Sample from .stages if not already done in custom.py
        super().__init__(steps=[SS_Sample()])