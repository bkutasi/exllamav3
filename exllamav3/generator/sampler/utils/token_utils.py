from dataclasses import dataclass, field
from typing import Dict, Set

import torch
import re
from functools import lru_cache
from exllamav3.tokenizer import Tokenizer
import logging
logger = logging.getLogger(__name__)

@dataclass
class NgramNode: # From ss_definitions.py lines 71-74
    value: int = 0
    children: Dict[int, 'NgramNode'] = field(default_factory=dict)

@lru_cache(maxsize=None)
def get_xtc_default_ignore_tokens(tokenizer: Tokenizer) -> Set[int]:
    logger.debug(f"get_xtc_default_ignore_tokens: Entered for tokenizer {type(tokenizer).__name__} with actual_vocab_size {tokenizer.actual_vocab_size} and config.vocab_size {tokenizer.config.vocab_size}")
    ignore_set = set()

    # Add standard special tokens if they exist and are valid IDs
    if tokenizer.bos_token_id is not None and 0 <= tokenizer.bos_token_id < tokenizer.config.vocab_size:
        ignore_set.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None and 0 <= tokenizer.eos_token_id < tokenizer.config.vocab_size:
        ignore_set.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None and 0 <= tokenizer.pad_token_id < tokenizer.config.vocab_size:
        ignore_set.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None and 0 <= tokenizer.unk_token_id < tokenizer.config.vocab_size:
        ignore_set.add(tokenizer.unk_token_id)

    logger.debug(f"get_xtc_default_ignore_tokens: Initial ignore set from special tokens: {len(ignore_set)} tokens.")

    try:
        # Get all token pieces. include_special_tokens=True should give us everything.
        # The length of this list should correspond to tokenizer.config.vocab_size
        logger.debug("get_xtc_default_ignore_tokens: Calling tokenizer.get_id_to_piece_list(include_special_tokens=True)...")
        all_token_pieces = tokenizer.get_id_to_piece_list(include_special_tokens=True)
        logger.debug(f"get_xtc_default_ignore_tokens: Received {len(all_token_pieces)} pieces from get_id_to_piece_list.")

        # Iterate up to actual_vocab_size as the original loop did,
        # but use the pre-fetched pieces.
        # This ensures we only consider tokens within the model's effective vocabulary range
        # for these general whitespace/replacement character rules.
        # Special tokens (BOS, EOS etc.) are handled above explicitly.
        # The original loop was `for i in range(tokenizer.actual_vocab_size):`
        
        # We should iterate up to len(all_token_pieces) if that list is comprehensive,
        # or tokenizer.config.vocab_size if that's the true upper bound for all valid token IDs.
        # The exllamav2 example iterates `for t in range(len(pieces))`.
        # Let's assume len(all_token_pieces) is the correct bound for iterating if include_special_tokens=True
        # covers everything.

        for i in range(len(all_token_pieces)):
            if i % 10000 == 0 and i > 0:
                logger.debug(f"get_xtc_default_ignore_tokens: Processing piece {i}/{len(all_token_pieces)}")

            if i in ignore_set: # Already added as a special token (BOS, EOS, etc.)
                continue
            
            token_str = all_token_pieces[i]

            if not isinstance(token_str, str):
                # This case should be rare if get_id_to_piece_list is robust
                logger.warning(f"get_xtc_default_ignore_tokens: Token ID {i} did not decode to a string (got {type(token_str)}). Adding to ignore set.")
                ignore_set.add(i)
                continue

            # Apply original checks
            if re.fullmatch(r"\s*\n\s*", token_str): # Matches tokens that are only a newline and optional surrounding whitespace
                ignore_set.add(i)
                continue
            
            # Matches tokens that consist of only whitespace but contain at least one double newline
            if "\n\n" in token_str and token_str.isspace(): # More precise: all characters are whitespace
                ignore_set.add(i)
                continue
            
            if "\ufffd" in token_str: # Replacement character
                ignore_set.add(i)
                continue
                
    except Exception as e:
        logger.error(f"get_xtc_default_ignore_tokens: Error during token processing: {e}. Results may be incomplete.", exc_info=True)
        # Fallback: return whatever has been collected so far, or just the initial special tokens.
        # This is safer than returning an empty set if the process fails midway.

    logger.debug(f"get_xtc_default_ignore_tokens: Finished. Found {len(ignore_set)} total tokens to ignore for tokenizer {type(tokenizer).__name__}.")
    return ignore_set

@lru_cache(maxsize=None) # From ss_definitions.py lines 114-143
def get_dry_default_sequence_breaker_tokens(tokenizer: Tokenizer) -> Set[int]:
    breakers = set()
    if tokenizer.bos_token_id is not None: breakers.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None: breakers.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None: breakers.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None: breakers.add(tokenizer.unk_token_id)

    for i in range(tokenizer.actual_vocab_size):
        if i in breakers:
            continue
            
        decoded_token_list = tokenizer.decode(torch.tensor([i]), decode_special_tokens=False)
        if not decoded_token_list : continue
        decoded_token = decoded_token_list[0]
        if decoded_token is None or not isinstance(decoded_token, str): continue
        
        if re.fullmatch(r"(\s*\n\s*){2,}", decoded_token): 
            breakers.add(i)
            continue
        
        # Consider any token that is all whitespace and contains at least one newline as a breaker
        # This is a bit broader than just double newlines.
        if decoded_token.isspace() and "\n" in decoded_token and not re.fullmatch(r"(\s*\n\s*){2,}", decoded_token): # Avoid double-adding if caught by above
             breakers.add(i)
             continue

        if "\ufffd" in decoded_token: # Replacement character
            breakers.add(i)
    return breakers