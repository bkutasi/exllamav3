import torch
from dataclasses import dataclass
from enum import Enum

class SS(Enum):
    INIT = 0  # only state.in_logits is valid
    DONE = 1  # finished, state.sample is valid
    LOGITS = 2  # state.logits is valid
    PROBS = 3  # state.probs is valid
    LOGITS_S = 4  # state.logits is valid, state.indices is valid
    PROBS_S = 5  # state.probs is valid but not normalized, indices are valid
    PROBS_N = 6  # state.probs is valid and normalized
    PROBS_N_S = 7  # state.probs is valid and normalized, indices are valid

@dataclass
class SamplingState:
    rand_u32: int
    bsz: int
    dim: int
    in_logits: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    sample: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    past_ids: torch.Tensor | None = None
    state: SS = SS.INIT

    def empty_sample(self):
        assert self.sample is None
        return torch.empty((self.bsz, 1), dtype = torch.long, device = self.in_logits.device)

    def empty_probs(self, reuse = True):
        if reuse and self.probs is not None:
            return self.probs
        return torch.empty((self.bsz, self.dim), dtype = torch.float, device = self.in_logits.device)

    def empty_logits(self, reuse = True):
        if reuse and self.logits is not None:
            return self.logits
        return torch.empty((self.bsz, self.dim), dtype = torch.float, device = self.in_logits.device)

class SS_Base:
    def run(self, state: SamplingState):
        raise NotImplementedError()
    def prep(self, in_state: SS):
        return None
    def alt(self):
        return None
    def reqs_past_ids(self):
        return False
