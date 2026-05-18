#!/usr/bin/env python3
import argparse
import asyncio
import json
import os

from transformers import AutoTokenizer


CEFR_ASPECTS = {
    "vocabulary": {
        "description": "vocabulary and word choices",
        "levels": {
            "A1": "very basic everyday words, limited range, repetitive simple vocabulary",
            "A2": "simple everyday vocabulary, basic but slightly wider range of common words",
            "B1": "adequate vocabulary for familiar topics, some paraphrasing when lacking words",
            "B2": "good range of vocabulary, can vary formulation, some precision and nuance",
            "C1": "wide range, precise vocabulary, good command of idiomatic expressions and collocations",
            "C2": "very rich, precise, nuanced vocabulary with excellent command of subtle distinctions",
        },
    },
    "syntax": {
        "description": "sentence structures and grammatical complexity",
        "levels": {
            "A1": "very short simple sentences, basic subject-verb-object only",
            "A2": "short sentences with basic coordination (and, but), limited patterns",
            "B1": "simple compound sentences with and/but/because, limited subordination",
            "B2": "varied structures, compound-complex sentences, clear clause embedding",
            "C1": "complex flexible syntax, wide range of structures with confident control",
            "C2": "sophisticated varied syntax with full structural control and elegant precision",
        },
    },
    "cohesion": {
        "description": "text organization, connectives, and discourse flow",
        "levels": {
            "A1": "simple connectors like and/then, minimal text organization",
            "A2": "basic linking words (and, but, because), simple sequencing",
            "B1": "common linking words (also, however, first/then), simple paragraph structure",
            "B2": "clear coherent text, variety of linking devices, logical argumentation flow",
            "C1": "well-structured text, sophisticated connectives, smooth natural transitions",
            "C2": "seamless organization, masterful rhetorical structure with effortless flow",
        },
    },
    "fluency": {
        "description": "naturalness, idiomatic usage, and native-like phrasing",
        "levels": {
            "A1": "stilted formulaic expressions, word-by-word construction, unnatural phrasing",
            "A2": "basic but somewhat formulaic, simple routine expressions",
            "B1": "reasonably smooth but with some unnatural or awkward phrasing",
            "B2": "generally natural, occasional awkward constructions, mostly fluent",
            "C1": "fluent idiomatic phrasing, natural collocations, spontaneous and confident",
            "C2": "effortlessly natural, subtle nuanced expression, native-like control of tone",
        },
    },
}

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

MAX_CONCURRENT = 30


def build_aspect_rewrite_prompt(source_text, aspect_name, aspect_info, target_level):
    level_desc = aspect_info["levels"][target_level]
    aspect_desc = aspect_info["description"]
    return (
        f"Rewrite the following text at CEFR {target_level} proficiency level, "
        f"with particular emphasis on the {aspect_desc}.\n\n"
        f"At CEFR {target_level}, the {aspect_desc} should be: {level_desc}\n\n"
        f"The rewritten text must clearly read as {target_level}-level English overall. "
        f"While focusing on {aspect_name}, ensure the entire text reflects {target_level} proficiency.\n\n"
        f"Output ONLY the rewritten text. No explanations.\n\n"
        f"Text: {source_text}"
    )


def build_retry_prompt(source_text, prev_rewrite, predicted_level, target_level,
                       aspect_name, aspect_info):
    level_desc = aspect_info["levels"][target_level]
    aspect_desc = aspect_info["description"]
    return (
        f"Your previous rewrite was classified as CEFR {predicted_level}, "
        f"but the target is {target_level}. The text needs to be more clearly "
        f"at {target_level} level.\n\n"
        f"Previous rewrite (classified as {predicted_level}):\n{prev_rewrite}\n\n"
        f"Please revise it to be unmistakably {target_level}-level, especially in "
        f"{aspect_desc}: {level_desc}\n\n"
        f"Make stronger changes to push the text to {target_level}. "
        f"Output ONLY the revised text. No explanations."
    )


def build_contrastive_prompt(source_text, aspect_name, aspect_info, level):
    level_desc = aspect_info["levels"][level]
    aspect_desc = aspect_info["description"]
    return (
        f"Rewrite the text below, changing only the {aspect_desc} "
        f"to CEFR {level} level. {level_desc}\n"
        f"Text: {source_text}"
    )


# ── vLLM backend ──────────────────────────────────────────────────────

def generate_vllm(source_texts, aspect_name, aspect_info, target_level, llm, sampling, rewrite_tokenizer):
    prompts = []
    for text in source_texts:
        msg = [
            {"role": "system", "content": "You are a careful English rewriting assistant."},
            {"role": "user", "content": build_aspect_rewrite_prompt(
                text, aspect_name, aspect_info, target_level
            )},
        ]
        prompts.append(rewrite_tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ))
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


def retry_vllm(source_texts, prev_rewrites, predicted_levels, aspect_name, aspect_info,
               target_level, llm, sampling, rewrite_tokenizer):
    from vllm import SamplingParams
    retry_sampling = SamplingParams(
        temperature=min(sampling.temperature + 0.4, 1.0),
        max_tokens=sampling.max_tokens,
    )
    prompts = []
    for src, prev, pred in zip(source_texts, prev_rewrites, predicted_levels):
        msg = [
            {"role": "system", "content": "You are a careful English rewriting assistant."},
            {"role": "user", "content": build_aspect_rewrite_prompt(
                src, aspect_name, aspect_info, target_level
            )},
            {"role": "assistant", "content": prev},
            {"role": "user", "content": build_retry_prompt(
                src, prev, pred, target_level, aspect_name, aspect_info
            )},
        ]
        prompts.append(rewrite_tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ))
    outputs = llm.generate(prompts, retry_sampling)
    return [o.outputs[0].text.strip() for o in outputs]


# ── OpenAI async backend ─────────────────────────────────────────────

async def _openai_rewrite_one(client, model, source_text, aspect_name, aspect_info, target_level, semaphore, temperature):
    user_content = build_aspect_rewrite_prompt(source_text, aspect_name, aspect_info, target_level)
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful English rewriting assistant."},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=512,
        )
    return response.choices[0].message.content.strip()


async def generate_openai_async(client, model, source_texts, aspect_name, aspect_info, target_level, temperature=0.3):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        _openai_rewrite_one(client, model, text, aspect_name, aspect_info, target_level, semaphore, temperature)
        for text in source_texts
    ]
    return await asyncio.gather(*tasks)


async def _openai_retry_one(client, model, source_text, prev_rewrite, predicted_level,
                            aspect_name, aspect_info, target_level, semaphore, temperature):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful English rewriting assistant."},
                {"role": "user", "content": build_aspect_rewrite_prompt(
                    source_text, aspect_name, aspect_info, target_level
                )},
                {"role": "assistant", "content": prev_rewrite},
                {"role": "user", "content": build_retry_prompt(
                    source_text, prev_rewrite, predicted_level, target_level,
                    aspect_name, aspect_info
                )},
            ],
            temperature=min(temperature + 0.4, 1.0),
            max_tokens=512,
        )
    return response.choices[0].message.content.strip()


async def retry_openai_async(client, model, source_texts, prev_rewrites, predicted_levels,
                             aspect_name, aspect_info, target_level, temperature=0.3):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        _openai_retry_one(client, model, src, prev, pred, aspect_name, aspect_info,
                          target_level, semaphore, temperature)
        for src, prev, pred in zip(source_texts, prev_rewrites, predicted_levels)
    ]
    return await asyncio.gather(*tasks)


# ── CEFR classifier validation ────────────────────────────────────────

async def validate_rewrites_with_classifier(
    cefr_classifier, rewrites, expected_level, source_texts,
    aspect_name, aspect_info, generate_fn, retry_fn,
    max_retries=8, tolerance=0, fresh_after=3,
):
    expected_idx = CEFR_ORDER.index(expected_level)
    result = list(rewrites)
    total = len(result)
    retried = 0
    fail_counts = [0] * total

    for retry_round in range(max_retries + 1):
        classifications = cefr_classifier.classify_batch([r for r in result])
        needs_retry_indices = []

        for i, cls in enumerate(classifications):
            pred_idx = CEFR_ORDER.index(cls["label"])
            if abs(pred_idx - expected_idx) > tolerance:
                needs_retry_indices.append(i)

        if not needs_retry_indices or retry_round == max_retries:
            break

        followup_indices = [i for i in needs_retry_indices if fail_counts[i] < fresh_after]
        fresh_indices = [i for i in needs_retry_indices if fail_counts[i] >= fresh_after]

        print(f"    Retry round {retry_round+1}: {len(needs_retry_indices)} mismatches "
              f"({len(followup_indices)} followup, {len(fresh_indices)} fresh)")

        if followup_indices:
            retry_sources = [source_texts[i] for i in followup_indices]
            retry_prev = [result[i] for i in followup_indices]
            retry_preds = [classifications[i]["label"] for i in followup_indices]
            new_rewrites = await retry_fn(
                retry_sources, retry_prev, retry_preds,
                aspect_name, aspect_info, expected_level,
            )
            for j, idx in enumerate(followup_indices):
                result[idx] = new_rewrites[j]

        if fresh_indices:
            fresh_sources = [source_texts[i] for i in fresh_indices]
            fresh_rewrites = await generate_fn(
                fresh_sources, aspect_name, aspect_info, expected_level,
            )
            for j, idx in enumerate(fresh_indices):
                result[idx] = fresh_rewrites[j]
                fail_counts[idx] = 0

        for idx in needs_retry_indices:
            fail_counts[idx] += 1
        retried += len(needs_retry_indices)

    # Final classification for stats
    final_cls = cefr_classifier.classify_batch(result)
    dist = {}
    for cls in final_cls:
        dist[cls["label"]] = dist.get(cls["label"], 0) + 1

    stats = {
        "expected": expected_level,
        "distribution": dist,
        "match_rate": sum(1 for c in final_cls if c["label"] == expected_level) / max(total, 1),
        "retried": retried,
    }
    return result, stats


# ── Build 4-tuples ────────────────────────────────────────────────────

def build_tuples(source_texts, tgt_rewrites, src_rewrites, aspect_name, aspect_info,
                 tgt_level, src_level, steering_tokenizer):
    tuples = []
    eos = steering_tokenizer.eos_token or ""
    for i, src_text in enumerate(source_texts):
        pos_user = build_contrastive_prompt(src_text, aspect_name, aspect_info, tgt_level)
        neg_user = build_contrastive_prompt(src_text, aspect_name, aspect_info, src_level)

        pos_messages = [{"role": "user", "content": pos_user}]
        neg_messages = [{"role": "user", "content": neg_user}]

        try:
            pos_prompt = steering_tokenizer.apply_chat_template(
                pos_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            neg_prompt = steering_tokenizer.apply_chat_template(
                neg_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            pos_prompt = steering_tokenizer.apply_chat_template(
                pos_messages, tokenize=False, add_generation_prompt=True
            )
            neg_prompt = steering_tokenizer.apply_chat_template(
                neg_messages, tokenize=False, add_generation_prompt=True
            )

        pos_full = pos_prompt + tgt_rewrites[i] + eos
        neg_full = neg_prompt + src_rewrites[i] + eos
        tuples.append([pos_prompt, pos_full, neg_prompt, neg_full])

    return tuples


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transitions', nargs='+', default=['B2_to_C1', 'B1_to_C1', 'A1_to_C1'])
    parser.add_argument('--aspects', nargs='+', default=list(CEFR_ASPECTS.keys()),
                        help='Aspects to generate (default: all)')
    parser.add_argument('--backend', choices=['vllm', 'openai'], default='vllm')
    parser.add_argument('--rewrite_model', default='unsloth/Llama-3.2-3B-Instruct')
    parser.add_argument('--steering_model', default='unsloth/Llama-3.2-3B-Instruct',
                        help='Steering model (for chat template in cache tuples)')
    parser.add_argument('--source_file', default='data/cefr/cefr_samples.json')
    parser.add_argument('--num_sources', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--gpu_memory', type=float, default=0.85)
    parser.add_argument('--out_dir', default='outputs/cefr')
    parser.add_argument('--max_retries', type=int, default=8,
                        help='Max retries for CEFR classifier validation')
    parser.add_argument('--tolerance', type=int, default=0,
                        help='CEFR level tolerance (0=exact match required)')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip CEFR classifier validation')
    parser.add_argument('--api_base', default='https://api.openai.com/v1',
                        help='OpenAI API base URL')
    args = parser.parse_args()

    with open(args.source_file) as f:
        cefr_samples = json.load(f)

    steering_tokenizer = AutoTokenizer.from_pretrained(args.steering_model)
    if steering_tokenizer.pad_token is None and steering_tokenizer.eos_token:
        steering_tokenizer.pad_token = steering_tokenizer.eos_token

    # Initialize backend
    llm = None
    sampling = None
    rewrite_tokenizer = None
    openai_client = None

    if args.backend == 'vllm':
        from dataclasses import asdict
        from vllm import LLM, SamplingParams
        from vllm.engine.arg_utils import EngineArgs

        rewrite_tokenizer = AutoTokenizer.from_pretrained(args.rewrite_model)
        engine_args = EngineArgs(
            model=args.rewrite_model,
            tokenizer=args.rewrite_model,
            gpu_memory_utilization=args.gpu_memory,
            max_model_len=2048,
        )
        llm = LLM(**asdict(engine_args))
        sampling = SamplingParams(temperature=args.temperature, max_tokens=512)
    else:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=args.api_base,
        )

    # Initialize CEFR classifier
    cefr_classifier = None
    if not args.skip_validation:
        import torch
        from transformers import AutoModelForSequenceClassification
        from cefr_experiment_run import CEFRClassifier
        cefr_classifier = CEFRClassifier(
            model_name_or_path="UniversalCEFR/xlm-roberta-base-cefr-all-classifier",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_length=512,
        )

    async def _generate_fn_vllm(texts, aspect_name, aspect_info, level):
        return generate_vllm(texts, aspect_name, aspect_info, level, llm, sampling, rewrite_tokenizer)

    async def _generate_fn_openai(texts, aspect_name, aspect_info, level):
        return await generate_openai_async(
            openai_client, args.rewrite_model, texts,
            aspect_name, aspect_info, level, args.temperature,
        )

    async def _retry_fn_vllm(sources, prev_rewrites, pred_levels, aspect_name, aspect_info, level):
        return retry_vllm(sources, prev_rewrites, pred_levels, aspect_name, aspect_info,
                          level, llm, sampling, rewrite_tokenizer)

    async def _retry_fn_openai(sources, prev_rewrites, pred_levels, aspect_name, aspect_info, level):
        return await retry_openai_async(
            openai_client, args.rewrite_model, sources, prev_rewrites, pred_levels,
            aspect_name, aspect_info, level, args.temperature,
        )

    generate_fn = _generate_fn_vllm if args.backend == 'vllm' else _generate_fn_openai
    retry_fn = _retry_fn_vllm if args.backend == 'vllm' else _retry_fn_openai

    os.makedirs(args.out_dir, exist_ok=True)

    for transition in args.transitions:
        src_level, tgt_level = transition.split('_to_')
        print(f"\n{'='*70}")
        print(f"Transition: {transition} (src={src_level}, tgt={tgt_level})")
        print(f"{'='*70}")

        source_texts = cefr_samples.get(src_level, [])[:args.num_sources]
        source_texts = [t for t in source_texts if len(t.split()) >= 5]
        print(f"  {len(source_texts)} source texts")

        for aspect_name in args.aspects:
            aspect_info = CEFR_ASPECTS[aspect_name]
            print(f"\n  --- Aspect: {aspect_name} ---")

            # Generate rewrites at target level
            print(f"  Generating {tgt_level} rewrites...")
            tgt_rewrites = await generate_fn(source_texts, aspect_name, aspect_info, tgt_level)

            # Generate rewrites at source level
            print(f"  Generating {src_level} rewrites...")
            src_rewrites = await generate_fn(source_texts, aspect_name, aspect_info, src_level)

            # Validate with CEFR classifier
            if cefr_classifier is not None:
                print(f"  Validating {tgt_level} rewrites...")
                tgt_rewrites, tgt_stats = await validate_rewrites_with_classifier(
                    cefr_classifier, tgt_rewrites, tgt_level, source_texts,
                    aspect_name, aspect_info, generate_fn, retry_fn,
                    max_retries=args.max_retries, tolerance=args.tolerance,
                )
                print(f"    {tgt_level} distribution: {tgt_stats['distribution']} "
                      f"(match={tgt_stats['match_rate']:.0%}, retried={tgt_stats['retried']})")

                print(f"  Validating {src_level} rewrites...")
                src_rewrites, src_stats = await validate_rewrites_with_classifier(
                    cefr_classifier, src_rewrites, src_level, source_texts,
                    aspect_name, aspect_info, generate_fn, retry_fn,
                    max_retries=args.max_retries, tolerance=args.tolerance,
                )
                print(f"    {src_level} distribution: {src_stats['distribution']} "
                      f"(match={src_stats['match_rate']:.0%}, retried={src_stats['retried']})")

            # Build 4-tuple cache entries
            tuples = build_tuples(
                source_texts, tgt_rewrites, src_rewrites,
                aspect_name, aspect_info, tgt_level, src_level,
                steering_tokenizer,
            )
            print(f"  Created {len(tuples)} contrastive pairs for {aspect_name}")

            if tuples:
                sample_pos = tuples[0][1][len(tuples[0][0]):]
                sample_neg = tuples[0][3][len(tuples[0][2]):]
                print(f"  Sample {tgt_level}: {sample_pos[:120]}...")
                print(f"  Sample {src_level}: {sample_neg[:120]}...")

            # Save per-aspect cache
            out_path = os.path.join(args.out_dir, f'tgt_caches_{aspect_name}_{transition}.json')
            cache = {transition: tuples}
            with open(out_path, 'w') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {out_path}")

if __name__ == '__main__':
    asyncio.run(main())
