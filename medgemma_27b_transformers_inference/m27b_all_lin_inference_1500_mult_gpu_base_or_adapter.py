"""One BF16 model per GPU; bounded image prefetch, batching, and ordered CSV merge."""
import argparse
import csv
import gc
import os
from pathlib import Path
import time
from transformers import BitsAndBytesConfig
ROOT = Path('/set/home/dir')
OUTPUT = ROOT / 'm27b_control_inf_transformers_max_tokens_1500.csv'
SYSTEM_PROMPT = {
    'role': 'system',
    'content': 'You are an expert clinician who specializes in genetics. You must respond with a logical rigor and deep explanatory depth.',
}
FIELDS = ['ppt_slide_number', 'question_index', 'prompt_number', 'patient_condition',
          'feature', 'short_answer_text', 'answer', 'phenotype_in_answer', 'confidence',
          'parsed_ok', 'error', 'latency_sec', 'image_path']


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--model-id', '--adapter', dest='model_id',
        default='google/medgemma-27b-it',
        help='Base model or PEFT adapter: Hugging Face repo ID or local directory',
    )
    p.add_argument(
        '--base-model', default=None,
        help='Override the base model recorded in an adapter config',
    )
    p.add_argument('--input-csv', type=Path, default='/set/input/prompts/all_prompts_gt.csv')
    p.add_argument('--image-root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=1500)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-merge-adapter', action='store_true')
    p.add_argument('--merge-results', action='store_true')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if min(a.batch_size, a.max_new_tokens, a.threads, a.workers) < 1:
        p.error('batch size, token limit, threads, and workers must be positive')
    return a


def merge_results(a):
    # Standard library only: merging never loads a model or initializes CUDA.
    with a.input_csv.open(newline='') as f:
        expected = sum(1 for _ in csv.DictReader(f))
    rows = {}
    for rank in range(a.workers):
        with (a.run_dir / f'rank{rank}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                i = int(row.pop('_source_row'))
                if i in rows:
                    raise ValueError(f'Duplicate source row {i}')
                rows[i] = row
    if set(rows) != set(range(expected)):
        raise ValueError('Worker outputs do not cover the input rows exactly')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows[i] for i in range(expected))
        os.replace(tmp, a.output)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'Saved {expected} rows to {a.output}', flush=True)


def worker(a):
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice
    import pandas as pd
    import torch
    from PIL import Image
    from huggingface_hub import file_exists
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != a.workers:
        raise ValueError(f'Expected {a.workers} workers; launcher started {world}')
    torch.set_num_threads(a.threads)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Worker {rank}: GPU {local_rank}, {torch.cuda.get_device_name(device)}', flush=True)

    def checkpoint_has_file(model_id, filename):
        path = Path(model_id).expanduser()
        if path.is_dir():
            return (path / filename).is_file()
        if path.is_absolute() or model_id.startswith(('./', '../', '~')):
            raise FileNotFoundError(f'Checkpoint directory not found: {model_id}')
        return file_exists(repo_id=model_id, filename=filename)

    model_id = a.model_id
    if Path(model_id).expanduser().is_dir():
        model_id = str(Path(model_id).expanduser().resolve())

    # Serialize checkpoint loading to limit simultaneous CPU RAM and disk pressure.
    # Models stay on their own GPUs after the lock is released.
    with (a.run_dir / 'loading.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        is_adapter = checkpoint_has_file(model_id, 'adapter_config.json')
        cfg = None

        if is_adapter:
            from peft import PeftConfig, PeftModel

            cfg = PeftConfig.from_pretrained(model_id)
            base_id = a.base_model or cfg.base_model_name_or_path
            if not base_id:
                raise ValueError(
                    'No base model recorded in adapter; supply --base-model'
                )
            if Path(base_id).expanduser().is_dir():
                base_id = str(Path(base_id).expanduser().resolve())
        else:
            if a.base_model is not None:
                raise ValueError(
                    '--base-model is only valid when --model-id is an adapter'
                )
            base_id = model_id

        if is_adapter:
            try:
                processor = AutoProcessor.from_pretrained(model_id)
            except OSError:
                processor = AutoProcessor.from_pretrained(base_id)
                print(
                    f'Worker {rank}: using base model processor',
                    flush=True,
                )
        else:
            processor = AutoProcessor.from_pretrained(base_id)
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token_id is None:
            if processor.tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer has neither a pad token nor an EOS token')
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        base = AutoModelForImageTextToText.from_pretrained(
            base_id,
            dtype=torch.bfloat16,
            attn_implementation='sdpa',
            quantization_config = bnb_config,
            device_map={'': local_rank},
            low_cpu_mem_usage=True,
        )
        model = base
        if is_adapter:
            model = PeftModel.from_pretrained(
                base,
                model_id,
                is_trainable=False,
                autocast_adapter_dtype=False,
            )
            if not a.no_merge_adapter:
                if str(cfg.peft_type).split('.')[-1] != 'LORA':
                    raise ValueError(
                        'Automatic merging expects LoRA; '
                        'use --no-merge-adapter'
                    )
                model = model.merge_and_unload()
        model.eval()
        # Retain checkpoint generation defaults, including sampling and stop tokens.
        if is_adapter and checkpoint_has_file(
            model_id, 'generation_config.json'
        ):
            model.generation_config = GenerationConfig.from_pretrained(
                model_id
            )
        model.generation_config.use_cache = True
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
        if hasattr(model.config, 'text_config'):
            model.config.text_config.use_cache = True
        del base
        gc.collect()
        torch.cuda.empty_cache()
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f'Worker {rank}: loaded; allocated={torch.cuda.memory_allocated(device)/2**30:.2f} GiB; '
          f'generation={model.generation_config}', flush=True)

    frame = pd.read_csv(a.input_csv)
    required = {'img_path', 'prompt', 'ppt_slide_number'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    jobs = [(i, frame.iloc[i].to_dict()) for i in range(rank, len(frame), world)]
    del frame

    def load_image(job):
        i, row = job
        img_root = "/data/houstoncm/Contradictions/"
        try:
            path =str( img_root + row['img_path'])
            with Image.open(path) as source:
                image = source.convert('RGB')
            return i, row, image, None
        except Exception as exc:
            return i, row, None, f'{type(exc).__name__}: {exc}'

    def generate(batch):
        messages = [[SYSTEM_PROMPT, {
            'role': 'user', 'content': [
                {'type': 'image', 'image': item[2]},
                {'type': 'text', 'text': f"{item[1]['prompt']} "},
            ],
        }] for item in batch]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, padding=True,
            return_dict=True, return_tensors='pt',
        ).to(device)
        # Cast floating image inputs only; token IDs and masks remain integers.
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                                 use_cache=True,
                                 pad_token_id=processor.tokenizer.pad_token_id)
        new_ids = ids[:, inputs['input_ids'].shape[1]:]
        answers = processor.batch_decode(new_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        return answers

    def result(item, answer, error, elapsed):
        i, row, _, _ = item
        return {
            '_source_row': i,
            'ppt_slide_number': row['ppt_slide_number'],
            'question_index': row.get('question_index', ''),
            'prompt_number': row.get('prompt_number', ''),
            'patient_condition': row.get('patient_condition', ''),
            'feature': row.get('feature', ''), 'short_answer_text': '',
            'answer': answer, 'phenotype_in_answer': None, 'confidence': '',
            'parsed_ok': answer is not None and answer.strip().lower() in {'yes', 'no'},
            'error': error, 'latency_sec': round(elapsed, 2),
            'image_path': row['img_path'],
        }

    batch_limit = a.batch_size

    def infer(batch):
        nonlocal batch_limit
        start = time.perf_counter()
        error = None
        oom = False
        try:
            answers = generate(batch)
        except torch.cuda.OutOfMemoryError as exc:
            error = f'{type(exc).__name__}: {exc}'
            oom = True
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        # Retry outside the exception block, so failed GPU tensors can be freed.
        if error is not None:
            if oom:
                gc.collect()
                torch.cuda.empty_cache()
                batch_limit = max(1, min(batch_limit, len(batch) // 2))
            if len(batch) > 1:
                print(f'Worker {rank}: splitting batch of {len(batch)}: {error[:200]}', flush=True)
                middle = len(batch) // 2
                return infer(batch[:middle]) + infer(batch[middle:])
            print(f'Worker {rank}: row {batch[0][0]} failed: {error[:300]}', flush=True)
            return [result(batch[0], None, error, time.perf_counter() - start)]
        elapsed = time.perf_counter() - start
        # For batching, latency is shared batch wall time, not per-row GPU time.
        return [result(item, answer, None, elapsed) for item, answer in zip(batch, answers)]

    start = time.perf_counter()
    completed = 0
    failed = 0
    with (a.run_dir / f'rank{rank}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['_source_row'] + FIELDS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Submit only one future batch ahead; do not decode all images into RAM.
            iterator = iter(jobs)
            def submit():
                chunk = list(islice(iterator, batch_limit))
                return pool.submit(lambda: [load_image(job) for job in chunk]) if chunk else None
            future = submit()
            while future is not None:
                batch = future.result()
                future = submit()
                try:
                    valid = [item for item in batch if item[3] is None]
                    output = [result(item, None, item[3], 0) for item in batch if item[3] is not None]
                    # A prefetched batch may exceed a newly reduced OOM limit.
                    current_limit = batch_limit
                    for offset in range(0, len(valid), current_limit):
                        output.extend(infer(valid[offset:offset + current_limit]))
                    writer.writerows(output)
                    f.flush()
                    completed += len(output)
                    failed += sum(bool(row['error']) for row in output)
                    print(f'Worker {rank}: {completed}/{len(jobs)} rows, {failed} errors, '
                          f'{time.perf_counter()-start:.1f}s, batch_limit={batch_limit}, '
                          f'peak={torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB', flush=True)
                finally:
                    for item in batch:
                        if item[2] is not None:
                            item[2].close()
    if jobs and failed == len(jobs):
        raise RuntimeError(f'All {failed} rows failed on worker {rank}; inspect log and worker CSV')


if __name__ == '__main__':
    args = arguments()
    if args.merge_results:
        merge_results(args)
    else:
        worker(args)
