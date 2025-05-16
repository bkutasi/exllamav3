import argparse
import json
import time
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import logging
from typing import Any, Dict, List, Optional, Set, Union

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)

# Add project root to path
script_path: str = os.path.abspath(__file__)
project_root: str = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, project_root)

try:
    from exllamav3 import model_init
    from exllamav3.models.model import Model
    from exllamav3.cache.cache import Cache
    from exllamav3.tokenizer.tokenizer import Tokenizer
    from exllamav3.generator.generator import Generator
    from exllamav3.generator.job import Job
    from exllamav3.generator.sampler.custom import CustomSampler
    from exllamav3.generator.sampler.stages import SS_Temperature, SS_TopP, SS_TopK, SS_Sample, SS_DRY, SS_XTC, SS_Smoothing, SS_Skew # MODIFIED
    from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant # For memory logging
    import torch
except ImportError as e:
    logger.error(f"Error: Required libraries not found. {e}")
    logger.error("Please ensure you have PyTorch installed and the exllamav3 library is accessible.")
    sys.exit(1)

# Global variables for model, tokenizer, generator
model: Optional[Model] = None
tokenizer: Optional[Tokenizer] = None
generator: Optional[Generator] = None
cache: Optional[Cache] = None
model_identifier: str = "exllamav3_kobold_model" # Default model identifier

def initialize_model(args: argparse.Namespace) -> None:
    global model, tokenizer, generator, model_identifier, cache

    logger.info(f"Initializing model: {args.model_dir}")
    # Pass quiet based on log_level or set to True for simplicity
    initialized_model, _, initialized_cache, initialized_tokenizer = model_init.init(args, quiet=True)
    model = initialized_model
    cache = initialized_cache
    tokenizer = initialized_tokenizer

    if hasattr(args, 'cache_quant') and args.cache_quant is not None:
        logger.debug(f" -- Cache quantization requested (args.cache_quant = {args.cache_quant}).")
    else:
        logger.debug(f" -- Using FP16 cache (cache_quant not specified or None).")

    logger.info("Model loaded.")
    logger.info("Tokenizer loaded.")
    if cache:
        logger.debug(f"Cache initialized by model_init with max_num_tokens = {cache.max_num_tokens}")
    else:
        logger.warning("Cache was not initialized by model_init.")

    if model and cache and tokenizer:
        generator = Generator(model, cache, tokenizer)
        logger.info("Generator created.")
        # Old memory logging removed from here
    else:
        logger.error("Failed to initialize generator due to missing model, cache, or tokenizer.")
        sys.exit(1)

    model_identifier = os.path.basename(os.path.normpath(args.model_dir))
    logger.info(f"Initialization complete. Model Identifier: {model_identifier}")

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Route HTTP access logs through our main logger at INFO level
        # Example: logger.info(""%s" %s %s", "GET /api/v1/model HTTP/1.1", "200", "-")
        logger.info(format, *args)

    def _send_response(self, status_code: int, content_type: str, body: str) -> None:
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))

    def _send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        self._send_response(status_code, 'application/json', json.dumps(data))

    def _send_error(self, status_code: int, message: str) -> None:
        error_data: Dict[str, Any] = {"error": {"message": message, "type": "api_error", "code": None}}
        self._send_json_response(status_code, error_data)

    def _send_sse_chunk(self, data: Dict[str, Any]) -> bool:
        """Sends a Server-Sent Event chunk with JSON data."""
        try:
            payload_bytes = f"event: message\ndata: {json.dumps(data)}\n\n".encode('utf-8')
            logger.debug(f"_send_sse_chunk: Attempting to write {len(payload_bytes)} bytes: {payload_bytes!r}")
            bytes_written = self.wfile.write(payload_bytes)
            logger.debug(f"_send_sse_chunk: self.wfile.write returned {bytes_written}.")
            self.wfile.flush()
            logger.debug("_send_sse_chunk: self.wfile.flush() called.")
            return True
        except BrokenPipeError:
            logger.warning("Client disconnected during SSE stream (BrokenPipeError).")
            return False
        except Exception as e_sse_send: # Catch any other error during send
            logger.error(f"Error in _send_sse_chunk: {e_sse_send}", exc_info=True)
            return False

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_POST(self) -> None:
        request_time_start: float = time.time() # Overall request timer
        global model, tokenizer, generator, cache # Ensure globals are accessible

        if self.path == '/api/v1/generate':
            if not model or not tokenizer or not generator or not cache:
                self._send_error(503, "Model is not initialized yet.")
                return

            # Performance Timing
            time_init_phase_start: float = 0.0
            time_init_phase_end_and_process_phase_start: float = 0.0
            time_process_phase_end_and_generate_phase_start: float = 0.0
            time_generate_phase_end: float = 0.0
            time_request_end: float = 0.0
            actual_completion_tokens: int = 0
            first_token_processed: bool = False # For logging logic consistency, not for event timing in non-streaming

            try:
                content_length_header = self.headers['Content-Length']
                if content_length_header is None:
                    self._send_error(411, "Content-Length header is required.")
                    return
                content_length: int = int(content_length_header)
                post_data: bytes = self.rfile.read(content_length)
                request_data: Dict[str, Any] = json.loads(post_data.decode('utf-8'))
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                self._send_error(400, f"Invalid JSON request: {e}")
                return
            except Exception as e:
                self._send_error(400, f"Error reading request: {e}")
                return

            prompt_str: Optional[str] = request_data.get("prompt")
            if prompt_str is None: # Allow empty prompt, but must be present
                self._send_error(400, "'prompt' field is required.")
                return

            memory_str: Optional[str] = request_data.get("memory")
            if memory_str:
                prompt_str = memory_str + prompt_str

            # Parameters
            max_new_tokens: int = request_data.get('max_length', 100)
            temperature: float = request_data.get('temperature', 0.8)
            top_p: float = request_data.get('top_p', 0.8)
            top_k: int = request_data.get('top_k', 40)
            stop_sequence_strs: List[str] = request_data.get('stop_sequence', [])

            # DRY Sampler Parameters
            dry_multiplier: float = request_data.get('dry_multiplier', 0.0)
            dry_base: float = request_data.get('dry_base', 1.75)
            dry_allowed_length: int = request_data.get('dry_allowed_length', 2)
            dry_sequence_breakers_str: List[str] = request_data.get('dry_sequence_breakers', [])
            dry_penalty_last_n: int = request_data.get('dry_penalty_last_n', 0) # Maps to range_val

            # XTC Sampler Parameters
            xtc_probability: float = request_data.get('xtc_probability', 0.0)
            xtc_threshold: float = request_data.get('xtc_threshold', 0.1)
            xtc_ignore_tokens_str: List[str] = request_data.get('xtc_ignore_tokens', [])

            # Temperature clamping
            min_temp: float = request_data.get('min_temp', 0.0)
            max_temp: float = request_data.get('max_temp', 0.0) # 0.0 means no max clamp

            # Smoothing Factor
            smoothing_factor: float = request_data.get('smoothing_factor', 0.0)

            # Skew
            skew: float = request_data.get('skew', 0.0)

            try:
                time_init_phase_start = time.time() # Start of init phase (prompt encoding, sampler setup)
                input_ids: torch.Tensor = tokenizer.encode(prompt_str)
                prompt_token_count: int = input_ids.shape[-1]
            except Exception as e:
                self._send_error(500, f"Error encoding prompt: {e}")
                return

            # Prepare stop conditions for Job constructor
            job_stop_conditions: List[Union[str, int]] = []
    
            # Add EOS token ID if it exists
            if tokenizer.eos_token_id is not None:
                job_stop_conditions.append(tokenizer.eos_token_id)
    
            # Add stop strings from the request
            for seq_str in stop_sequence_strs:
                if seq_str:
                    job_stop_conditions.append(seq_str)
    
            final_job_stop_conditions = job_stop_conditions if job_stop_conditions else None
            
            # Sampler Configuration
            sampler_steps: List[Any] = []

            # Temperature, Min/Max Temp
            if not (temperature == 1.0 and min_temp == 0.0 and max_temp == 0.0):
                log_msg = f"SS_Temperature active: temp={temperature}"
                if min_temp > 0.0: log_msg += f", min_temp={min_temp}"
                if max_temp > 0.0: log_msg += f", max_temp={max_temp}"
                logger.debug(log_msg)
                sampler_steps.append(SS_Temperature(temperature, min_temp=min_temp, max_temp=max_temp))
            else:
                logger.debug("SS_Temperature not active (temp=1.0, no min/max clamp).")

            # Smoothing Factor (after temperature)
            if smoothing_factor > 0.0:
                logger.debug(f"Smoothing Sampler active: factor={smoothing_factor}")
                sampler_steps.append(SS_Smoothing(factor=smoothing_factor))

            # DRY operates on logits, so apply it after temperature and smoothing.
            if dry_multiplier > 0.0:
                tokenized_breakers_set: Set[int] = set()
                for breaker_str in dry_sequence_breakers_str:
                    if breaker_str:
                        breaker_ids = tokenizer.encode(breaker_str, add_bos=False, add_eos=False)[0].tolist()
                        if breaker_ids:
                            tokenized_breakers_set.update(breaker_ids)
                
                final_tokenized_breakers_set = tokenized_breakers_set if tokenized_breakers_set else None

                logger.debug(f"DRY Sampler active: mult={dry_multiplier}, base={dry_base}, allow_len={dry_allowed_length}, range={dry_penalty_last_n}, breakers={final_tokenized_breakers_set}")
                sampler_steps.append(SS_DRY(multiplier=dry_multiplier,
                                            base=dry_base,
                                            allowed_length=dry_allowed_length,
                                            sequence_breakers=final_tokenized_breakers_set,
                                            range_val=dry_penalty_last_n))

            # XTC Sampler (after temp, smoothing, DRY, before top_k/top_p)
            if xtc_probability > 0.0:
                tokenized_xtc_ignore_set: Set[int] = set()
                if tokenizer: # Ensure tokenizer is available
                    for token_str_xtc in xtc_ignore_tokens_str:
                        if token_str_xtc:
                            try:
                                token_ids_xtc = tokenizer.encode(token_str_xtc, add_bos=False, add_eos=False)[0].tolist()
                                if token_ids_xtc:
                                    tokenized_xtc_ignore_set.update(token_ids_xtc)
                            except Exception as e_xtc_token:
                                logger.warning(f"Could not tokenize XTC ignore token string '{token_str_xtc}': {e_xtc_token}")
                
                final_xtc_ignore_set = tokenized_xtc_ignore_set if tokenized_xtc_ignore_set else None
                logger.debug(f"XTC Sampler active: probability={xtc_probability}, threshold={xtc_threshold}, ignore_tokens_count={len(final_xtc_ignore_set) if final_xtc_ignore_set else 0}")
                sampler_steps.append(SS_XTC(probability=xtc_probability,
                                            threshold=xtc_threshold,
                                            ignore_tokens=final_xtc_ignore_set))
            else:
                logger.debug("XTC Sampler not active (xtc_probability is 0.0 or not set).")

            # Skew Sampler (after XTC, before TopK/TopP)
            if abs(skew) > 1e-6: # If skew is non-zero
                logger.debug(f"Skew Sampler active: skew={skew}")
                sampler_steps.append(SS_Skew(skew=skew))

            # Standard sampling settings after DRY, XTC, Skew
            if top_k > 0:
                sampler_steps.append(SS_TopK(top_k))
            if 0.0 < top_p < 1.0: 
                sampler_steps.append(SS_TopP(top_p))

            sampler_steps.append(SS_Sample())
            custom_sampler: CustomSampler = CustomSampler(sampler_steps)

            # Generation
            time_init_phase_end_and_process_phase_start = time.time() # End of init (prompt encoding, sampler setup), start of process (Job object creation)
            # generation_start_time = time.time() # This will be replaced by more granular timing
            try:
                job = Job(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    sampler=custom_sampler,
                    stop_conditions=final_job_stop_conditions,
                    decode_special_tokens=False, 
                    completion_only=True 
                )
                generator.enqueue(job)
                time_process_phase_end_and_generate_phase_start = time.time() # End of process (Job obj + enqueue), start of generate (iterate loop)

                job_done = False
                last_results_dict = {} 

                while generator.num_remaining_jobs() > 0 and not job_done:
                    results_list: List[Dict[str, Any]] = generator.iterate()
                    for r_item in results_list:
                        if r_item["serial"] != job.serial_number:
                            continue

                        if r_item["eos"]:
                            last_results_dict = r_item 
                            job_done = True
                            break 
                    if job_done:
                        break
                
                time_generate_phase_end = time.time() # End of generation loop
                actual_completion_tokens = last_results_dict.get("new_tokens", 0) if last_results_dict else 0
                # generation_time = time.time() - generation_start_time # Replaced by new calculation
                output_str = job.full_completion if job.full_completion is not None else ""

                response_data: Dict[str, Any] = {
                    "results": [{
                        "text": output_str.strip(),
                        "finish_reason": last_results_dict.get("eos_reason", "unknown") if last_results_dict else "unknown",
                        "logprobs": None, 
                        "prompt_tokens": prompt_token_count,
                        "completion_tokens": last_results_dict.get("new_tokens", 0) if last_results_dict else 0
                    }]
                }
                self._send_json_response(200, response_data)
                
                # output_token_count = last_results_dict.get("new_tokens", 0) if last_results_dict else tokenizer.num_tokens(output_str) # actual_completion_tokens is used now
                # tokens_per_sec = output_token_count / generation_time if generation_time > 0 else 0 # Replaced by new calculation
                # logger.info(f"Generated {output_token_count} tokens ({len(output_str)} chars) in {generation_time:.2f}s ({tokens_per_sec:.2f} t/s). Prompt: {prompt_token_count} tokens. Max new: {max_new_tokens}.")

                # Performance Logging
                time_request_end = time.time()
                init_s: float = (time_init_phase_end_and_process_phase_start - time_init_phase_start) if time_init_phase_start > 0 and time_init_phase_end_and_process_phase_start > time_init_phase_start else 0.0
                
                # For non-streaming:
                # process_s is Job creation + enqueue time (from time_init_phase_end_and_process_phase_start to time_process_phase_end_and_generate_phase_start)
                # generate_s is the iterate loop time (from time_process_phase_end_and_generate_phase_start to time_generate_phase_end)
                process_s: float = (time_process_phase_end_and_generate_phase_start - time_init_phase_end_and_process_phase_start) if time_init_phase_end_and_process_phase_start > 0 and time_process_phase_end_and_generate_phase_start > time_init_phase_end_and_process_phase_start else 0.0
                generate_s: float = (time_generate_phase_end - time_process_phase_end_and_generate_phase_start) if time_process_phase_end_and_generate_phase_start > 0 and time_generate_phase_end > time_process_phase_end_and_generate_phase_start else 0.0

                total_s: float = (time_request_end - request_time_start) if request_time_start > 0 and time_request_end > request_time_start else 0.0

                ctx_limit_val: int = cache.max_num_tokens if cache else 0
                prompt_len: int = prompt_token_count # prompt_token_count is already defined
                max_new_tokens_req: int = max_new_tokens # max_new_tokens is 'max_length' from kobold request

                process_tps: float = prompt_len / process_s if process_s > 0.001 else 0.0
                generate_tps: float = actual_completion_tokens / generate_s if generate_s > 0.001 else 0.0

                logger.info(f"CtxLimit:{prompt_len}/{ctx_limit_val}, Amt:{actual_completion_tokens}/{max_new_tokens_req}, Init:{init_s:.2f}s, Process:{process_s:.2f}s ({process_tps:.2f}T/s), Generate:{generate_s:.2f}s ({generate_tps:.2f}T/s), Total:{total_s:.2f}s")

            except Exception as e:
                logger.error(f"Error during generation: {e}", exc_info=True)
                self._send_error(500, f"Error during generation: {e}")
                return
        elif self.path == '/api/extra/generate/stream':
            if not model or not tokenizer or not generator or not cache:
                self._send_error(503, "Model is not initialized yet.")
                return

            # Performance Timing
            time_init_phase_start: float = 0.0
            time_init_phase_end_and_process_phase_start: float = 0.0
            time_process_phase_end_and_generate_phase_start: float = 0.0
            time_generate_phase_end: float = 0.0
            time_request_end: float = 0.0
            actual_completion_tokens: int = 0
            first_token_processed: bool = False

            try:
                content_length_header = self.headers['Content-Length']
                if content_length_header is None:
                    self._send_error(411, "Content-Length header is required.")
                    return
                content_length: int = int(content_length_header)
                post_data: bytes = self.rfile.read(content_length)
                request_data: Dict[str, Any] = json.loads(post_data.decode('utf-8'))
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                self._send_error(400, f"Invalid JSON request: {e}")
                return
            except Exception as e:
                self._send_error(400, f"Error reading request: {e}")
                return

            prompt_str: Optional[str] = request_data.get("prompt")
            if prompt_str is None:
                self._send_error(400, "'prompt' field is required.")
                return

            memory_str: Optional[str] = request_data.get("memory")
            if memory_str:
                prompt_str = memory_str + prompt_str

            max_new_tokens: int = request_data.get('max_length', 100)
            temperature: float = request_data.get('temperature', 0.8)
            top_p: float = request_data.get('top_p', 0.8)
            top_k: int = request_data.get('top_k', 40)
            stop_sequence_strs: List[str] = request_data.get('stop_sequence', [])
            dry_multiplier: float = request_data.get('dry_multiplier', 0.0)
            dry_base: float = request_data.get('dry_base', 1.75)
            dry_allowed_length: int = request_data.get('dry_allowed_length', 2)
            dry_sequence_breakers_str: List[str] = request_data.get('dry_sequence_breakers', [])
            dry_penalty_last_n: int = request_data.get('dry_penalty_last_n', 0)
            xtc_probability: float = request_data.get('xtc_probability', 0.0)
            xtc_threshold: float = request_data.get('xtc_threshold', 0.1)
            xtc_ignore_tokens_str: List[str] = request_data.get('xtc_ignore_tokens', [])
            
            # Temperature clamping
            min_temp: float = request_data.get('min_temp', 0.0)
            max_temp: float = request_data.get('max_temp', 0.0) # 0.0 means no max clamp

            # Smoothing Factor
            smoothing_factor: float = request_data.get('smoothing_factor', 0.0)

            # Skew
            skew: float = request_data.get('skew', 0.0)

            try:
                time_init_phase_start = time.time() # Start of init phase (prompt encoding, sampler setup)
                input_ids: torch.Tensor = tokenizer.encode(prompt_str)
                prompt_token_count: int = input_ids.shape[-1]
            except Exception as e:
                self._send_error(500, f"Error encoding prompt: {e}")
                return

            job_stop_conditions: List[Union[str, int]] = []
            if tokenizer.eos_token_id is not None:
                job_stop_conditions.append(tokenizer.eos_token_id)
            for seq_str in stop_sequence_strs:
                if seq_str:
                    job_stop_conditions.append(seq_str)
            final_job_stop_conditions = job_stop_conditions if job_stop_conditions else None

            sampler_steps: List[Any] = []

            # Temperature, Min/Max Temp
            if not (temperature == 1.0 and min_temp == 0.0 and max_temp == 0.0):
                log_msg = f"SS_Temperature active (stream): temp={temperature}"
                if min_temp > 0.0: log_msg += f", min_temp={min_temp}"
                if max_temp > 0.0: log_msg += f", max_temp={max_temp}"
                logger.debug(log_msg)
                sampler_steps.append(SS_Temperature(temperature, min_temp=min_temp, max_temp=max_temp))
            else:
                logger.debug("SS_Temperature not active for stream (temp=1.0, no min/max clamp).")

            # Smoothing Factor (after temperature)
            if smoothing_factor > 0.0:
                logger.debug(f"Smoothing Sampler active (stream): factor={smoothing_factor}")
                sampler_steps.append(SS_Smoothing(factor=smoothing_factor))

            if dry_multiplier > 0.0:
                tokenized_breakers_set: Set[int] = set()
                for breaker_str in dry_sequence_breakers_str:
                    if breaker_str:
                        breaker_ids = tokenizer.encode(breaker_str, add_bos=False, add_eos=False)[0].tolist()
                        if breaker_ids:
                            tokenized_breakers_set.update(breaker_ids)
                final_tokenized_breakers_set = tokenized_breakers_set if tokenized_breakers_set else None
                logger.debug(f"DRY Sampler active (stream): mult={dry_multiplier}, base={dry_base}, allow_len={dry_allowed_length}, range={dry_penalty_last_n}, breakers={final_tokenized_breakers_set}")
                sampler_steps.append(SS_DRY(multiplier=dry_multiplier,
                                            base=dry_base,
                                            allowed_length=dry_allowed_length,
                                            sequence_breakers=final_tokenized_breakers_set,
                                            range_val=dry_penalty_last_n))
            else:
                logger.debug("DRY Sampler not active for stream (dry_multiplier is 0.0 or not set).")

            if xtc_probability > 0.0:
                tokenized_xtc_ignore_set: Set[int] = set()
                if tokenizer: 
                    for token_str_xtc in xtc_ignore_tokens_str:
                        if token_str_xtc:
                            try:
                                token_ids_xtc = tokenizer.encode(token_str_xtc, add_bos=False, add_eos=False)[0].tolist()
                                if token_ids_xtc:
                                    tokenized_xtc_ignore_set.update(token_ids_xtc)
                            except Exception as e_xtc_token:
                                logger.warning(f"Could not tokenize XTC ignore token string '{token_str_xtc}': {e_xtc_token}")
                final_xtc_ignore_set = tokenized_xtc_ignore_set if tokenized_xtc_ignore_set else None
                logger.debug(f"XTC Sampler active (stream): probability={xtc_probability}, threshold={xtc_threshold}, ignore_tokens_count={len(final_xtc_ignore_set) if final_xtc_ignore_set else 0}")
                sampler_steps.append(SS_XTC(probability=xtc_probability,
                                            threshold=xtc_threshold,
                                            ignore_tokens=final_xtc_ignore_set))
            else:
                logger.debug("XTC Sampler not active for stream (xtc_probability is 0.0 or not set).")

            # Skew Sampler (after XTC, before TopK/TopP)
            if abs(skew) > 1e-6: # If skew is non-zero
                logger.debug(f"Skew Sampler active (stream): skew={skew}")
                sampler_steps.append(SS_Skew(skew=skew))

            if top_k > 0:
                logger.debug(f"Applying SS_TopK (stream): {top_k}")
                sampler_steps.append(SS_TopK(top_k))
            if 0.0 < top_p < 1.0: 
                logger.debug(f"Applying SS_TopP (stream): {top_p}")
                sampler_steps.append(SS_TopP(top_p))
            
            logger.debug("Applying SS_Sample (stream).")
            sampler_steps.append(SS_Sample())
            custom_sampler: CustomSampler = CustomSampler(sampler_steps)

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close') 
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            job_done: bool = False
            job: Optional[Job] = None
            # stream_completion_tokens: int = 0 # Will use actual_completion_tokens
            finish_reason: Optional[str] = None
            # generation_start_time = time.time() # Replaced by granular timing

            try:
                time_init_phase_end_and_process_phase_start = time.time() # End of init (prompt, sampler), start of process (Job object creation + enqueue + time to first token)
                logger.debug("Debugging stop_conditions types (stream):")
                if final_job_stop_conditions is not None:
                    for i, item in enumerate(final_job_stop_conditions):
                        logger.debug(f"  stop_conditions[{i}] type: {type(item)}")
                else:
                    logger.debug("  stop_conditions is None (stream)")
                job = Job(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    sampler=custom_sampler, 
                    stop_conditions=final_job_stop_conditions,
                    decode_special_tokens=False 
                )
                generator.enqueue(job)

                logger.debug("Kobold Stream: Entering generation loop.")
                while generator.num_remaining_jobs() > 0 and not job_done:
                    logger.debug(f"Kobold Stream: Top of iterate loop. Remaining jobs: {generator.num_remaining_jobs()}, job_done: {job_done}")
                    results: List[Dict[str, Any]] = generator.iterate()
                    for r in results:
                        logger.debug(f"Kobold Stream: Processing result: {r}")
                        if job and r["serial"] != job.serial_number:
                            continue

                        if r["stage"] == "streaming":
                            chunk: str = r.get("text", "")
                            if chunk:
                                if not first_token_processed:
                                    time_process_phase_end_and_generate_phase_start = time.time() # End of process, start of token generation
                                    first_token_processed = True
                                logger.debug(f"Kobold Stream: Sending chunk: {chunk}")
                                sse_data: Dict[str, Any] = {"token": chunk}
                                if not self._send_sse_chunk(sse_data):
                                    logger.warning("Client disconnected, cancelling job.")
                                    if job: generator.cancel(job)
                                    job_done = True
                                    break 
                        if r["eos"]:
                            logger.debug(f"Kobold Stream: EOS received. Reason: {r.get('eos_reason', 'N/A')}, New Tokens: {r.get('new_tokens', 'N/A')}")
                            if not first_token_processed: # If EOS is the first event (e.g. empty generation)
                                time_process_phase_end_and_generate_phase_start = time.time()
                                first_token_processed = True
                            time_generate_phase_end = time.time() # End of token generation
                            actual_completion_tokens = r.get("new_tokens", 0)
                            finish_reason = r.get("eos_reason", "stop")
                            job_done = True
                            break
                    if job_done:
                        break

                # Ensure time_generate_phase_end is set if loop exited for other reasons
                if not time_generate_phase_end and job_done: # job_done might be true from client disconnect
                    time_generate_phase_end = time.time()
                if not job_done: # If loop finished due to num_remaining_jobs == 0 but not EOS
                    time_generate_phase_end = time.time() # Mark end of generation attempt
                
                if not first_token_processed and (job_done or generator.num_remaining_jobs() == 0) : # If disconnected/ended before any token
                     time_process_phase_end_and_generate_phase_start = time_generate_phase_end # Process took all the time up to this point
                     first_token_processed = True # Allow logging calculations to proceed, assuming 0 generated tokens

                # generation_time = time.time() - generation_start_time # Replaced by new calculation

                logger.debug(f"Kobold Stream: Exited generation loop. Final finish_reason: {finish_reason}, Stream completion tokens: {actual_completion_tokens}")
                final_sse_data: Dict[str, Any] = {
                    "status": "finished",
                    "stop_reason": finish_reason if finish_reason else "length", 
                    "text": "" 
                }
                if self._send_sse_chunk(final_sse_data):
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    logger.debug("Kobold Stream: Sent [DONE] marker.")

                # output_token_count = actual_completion_tokens # actual_completion_tokens is used directly
                # tokens_per_sec = output_token_count / generation_time if generation_time > 0 else 0 # Replaced by new calculation
                # logger.info(f"Streaming generated {output_token_count} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} t/s). Prompt: {prompt_token_count} tokens. Max new: {max_new_tokens}.")

                # Performance Logging
                time_request_end = time.time()
                init_s: float = (time_init_phase_end_and_process_phase_start - time_init_phase_start) if time_init_phase_start > 0 and time_init_phase_end_and_process_phase_start > time_init_phase_start else 0.0
                
                if first_token_processed: # True for streaming once first token is out
                    process_s: float = (time_process_phase_end_and_generate_phase_start - time_init_phase_end_and_process_phase_start) if time_init_phase_end_and_process_phase_start > 0 and time_process_phase_end_and_generate_phase_start > time_init_phase_end_and_process_phase_start else 0.0
                    generate_s: float = (time_generate_phase_end - time_process_phase_end_and_generate_phase_start) if time_process_phase_end_and_generate_phase_start > 0 and time_generate_phase_end > time_process_phase_end_and_generate_phase_start else 0.0
                else: # Fallback for streaming if no tokens were ever generated/streamed (e.g. immediate EOS or error/disconnect before first token)
                    # In this case, the "process" phase effectively ran until the end of what we could measure for generation.
                    process_s: float = (time_generate_phase_end - time_init_phase_end_and_process_phase_start) if time_init_phase_end_and_process_phase_start > 0 and time_generate_phase_end > time_init_phase_end_and_process_phase_start else 0.0
                    generate_s: float = 0.0

                total_s: float = (time_request_end - request_time_start) if request_time_start > 0 and time_request_end > request_time_start else 0.0

                ctx_limit_val: int = cache.max_num_tokens if cache else 0
                prompt_len: int = prompt_token_count # prompt_token_count is already defined
                max_new_tokens_req: int = max_new_tokens # max_new_tokens is 'max_length' from kobold request

                process_tps: float = prompt_len / process_s if process_s > 0.001 else 0.0
                generate_tps: float = actual_completion_tokens / generate_s if generate_s > 0.001 else 0.0

                logger.info(f"CtxLimit:{prompt_len}/{ctx_limit_val}, Amt:{actual_completion_tokens}/{max_new_tokens_req}, Init:{init_s:.2f}s, Process:{process_s:.2f}s ({process_tps:.2f}T/s), Generate:{generate_s:.2f}s ({generate_tps:.2f}T/s), Total:{total_s:.2f}s")

            except BrokenPipeError:
                 logger.warning("Client disconnected during SSE stream.")
            except Exception as e:
                logger.error(f"Error during streaming generation: {e}", exc_info=True)
                try:
                    error_sse_data: Dict[str, Any] = {"status": "error", "message": f"Generation error: {e}"}
                    self._send_sse_chunk(error_sse_data)
                except:
                    pass 
                finally:
                    if job and not job_done:
                        generator.cancel(job)
        else:
            self._send_error(404, "Not Found. Use POST /api/v1/generate or POST /api/extra/generate/stream")

    def do_GET(self) -> None:
        global model, model_identifier # Ensure model_identifier is accessible

        if self.path == '/api/v1/model':
            if not model: # Check if the main model object is initialized
                self._send_error(503, "Model is not initialized yet.")
                return # Important to return after handling
            
            # Ensure model_identifier has been set (it's set at the end of initialize_model)
            if not model_identifier: # Should ideally not happen if model is initialized
                 self._send_error(503, "Model identifier not available.")
                 return # Important to return after handling

            response_data = {
                "result": {
                    "result": model_identifier
                }
            }
            self._send_json_response(200, response_data)
            return # Important to return after handling
        elif self.path == '/':
            self._send_response(200, 'text/plain', 'ExLlamaV3 Kobold KCPP API Server is running. Use POST /api/v1/generate.')
            return # Important to return
        else:
            self._send_error(404, "Not Found.")
            return # Important to return

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True

def main() -> None:
    global model, tokenizer, generator, cache

    parser = argparse.ArgumentParser(description="ExLlamaV3 Kobold KCPP Compatible API Endpoint")
    # Model init arguments
    model_init.add_args(parser, cache=True) # Adds --model_dir, --cache_size, etc.

    # Server arguments
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port to listen on (Default: 5001)")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host to bind to (Default: 0.0.0.0)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")

    args = parser.parse_args()

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level_str: str = args.log_level.upper()
    log_level_val: int = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level_val, format=log_format, force=True)
    logger.info("Starting Kobold KCPP API endpoint...")

    initialize_model(args)

    if not model or not tokenizer or not generator:
        logger.error("Failed to initialize model, tokenizer, or generator. Exiting.")
        sys.exit(1)

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, APIHandler)

    # --- Log Memory Usage and VRAM Info per Device ---
    # Note: Model memory per device is not easily available from model.get_storage_info()
    # The log below shows total model memory and per-device cache/VRAM.
    # A more accurate log would require iterating through model layers/components.

    if model: # Ensure model is initialized
        model_bits, _, model_vram_bits = model.get_storage_info()
        model_bytes = model_vram_bits / 8
        model_mb = model_bytes / (1024 * 1024)
        logger.info(f"Total Model Memory Usage: {model_mb:.2f} MB")

    cache_bytes_per_device: Dict[int, int] = {}
    if cache: # Ensure cache object exists
        for layer in cache.layers:
            device_idx = layer.device.index # Get device index for the layer
            if device_idx not in cache_bytes_per_device:
                cache_bytes_per_device[device_idx] = 0

            if isinstance(layer, CacheLayer_fp16):
                if layer.k is not None: cache_bytes_per_device[device_idx] += layer.k.nbytes
                if layer.v is not None: cache_bytes_per_device[device_idx] += layer.v.nbytes
            elif isinstance(layer, CacheLayer_quant):
                if layer.qk is not None: cache_bytes_per_device[device_idx] += layer.qk.nbytes
                if layer.qv is not None: cache_bytes_per_device[device_idx] += layer.qv.nbytes
                if layer.sk is not None: cache_bytes_per_device[device_idx] += layer.sk.nbytes
                if layer.sv is not None: cache_bytes_per_device[device_idx] += layer.sv.nbytes

    try:
        if torch.cuda.is_available():
            # Get unique device indices from cache layers
            device_indices = sorted(list(cache_bytes_per_device.keys()))

            if device_indices:
                for device_idx in device_indices:
                    device_name: str = torch.cuda.get_device_name(device_idx)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
                    free_mb: float = free_bytes / (1024 * 1024)
                    total_mb: float = total_bytes / (1024 * 1024)

                    # Include cache memory for this device in the log
                    cache_mb_this_device = cache_bytes_per_device.get(device_idx, 0) / (1024 * 1024)

                    logger.info(f"Device {device_idx} ({device_name}): VRAM Free: {free_mb:.2f} MB / Total: {total_mb:.2f} MB, Cache: {cache_mb_this_device:.2f} MB")
            else:
                logger.info("No CUDA devices found in cache layers, skipping per-device VRAM/Cache info logging.")
        else:
            logger.info("CUDA not available, skipping VRAM info logging.")
    except AttributeError as e:
        logger.warning(f"Could not retrieve VRAM info: {e}. This might happen if the loaded model object does not have a 'device' attribute, possibly due to the model type or an issue during initialization.")
    except Exception as e:
        logger.warning(f"An error occurred while logging VRAM info: {e}")
    # --- End Log Memory Usage and VRAM Info per Device ---

    logger.info(f"Serving on http://{args.host}:{args.port}/api/v1/generate")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
    finally:
        if 'httpd' in locals() and httpd:
            httpd.server_close() # Ensure sockets are closed
        logger.info("Server stopped.")
        
if __name__ == "__main__":
    main()