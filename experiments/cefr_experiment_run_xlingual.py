from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import time
import torch
from typing import List, Any, Dict, Union
import numpy as np
from control_block import WrappedReadingVecModel


model_name = 'unsloth/Llama-3.2-3B-Instruct'
cefr_classifier_name = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_ID = {level: i for i, level in enumerate(CEFR_ORDER)}

LANG_NAMES = {
    'de': 'German',
    'it': 'Italian',
    'en': 'English',
}


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
model.eval()


class CEFRClassifier:
    def __init__(self, model_name_or_path: str, device: str = "cuda:0", max_length: int = 512):
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

        raw_id2label = getattr(self.model.config, "id2label", None)
        if raw_id2label:
            self.id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            self.id2label = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}

    @torch.inference_mode()
    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        results = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            for i, pred_id in enumerate(pred_ids):
                pred_id_int = int(pred_id.item())
                results.append({
                    "label": self.id2label[pred_id_int],
                    "confidence": float(probs[i, pred_id_int].item()),
                    "probs": {self.id2label[j]: float(probs[i, j].item()) for j in range(probs.shape[-1])},
                })
        return results

    def classify_one(self, text: str) -> Dict[str, Any]:
        return self.classify_batch([text], batch_size=1)[0]


def masked_mean(hidden_states, mask):
    mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def last_token_pool(hidden_states, mask):
    last_idx = mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]


def get_hidden_states(outputs, attention_mask, hidden_layers, pooling='mean'):
    hidden_states_layers = {}
    for layer in hidden_layers:
        hidden_states = outputs['hidden_states'][layer]
        if pooling == 'last_token':
            hidden_states = last_token_pool(hidden_states, attention_mask)
        else:
            hidden_states = masked_mean(hidden_states, attention_mask)
        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        hidden_states_layers[layer] = hidden_states.detach().cpu()
    return hidden_states_layers


def get_rep_directions(hidden_states, hidden_layers, n_components: int = 1, use_pca: bool = True, normalize: bool = True):
    directions = {}
    raw_norms = {}
    for layer in hidden_layers:
        H_train = hidden_states[layer]
        H_train_mean = H_train.mean(axis=0, keepdims=True).cpu().detach()
        if use_pca:
            H_np = H_train.cpu().detach().numpy().astype(np.float32)
            if H_np.ndim == 1:
                H_np = H_np[np.newaxis, :]
            U, s, Vt = np.linalg.svd(H_np, full_matrices=False)
            direction = torch.from_numpy(Vt[:n_components].copy())
            sign_check = (H_train_mean.cpu().float() @ direction[0]).item()
            if sign_check < 0:
                direction = -direction
        else:
            direction = H_train_mean
        raw_norms[layer] = float(direction.norm().item())
        if normalize and direction.norm() > 1e-8:
            direction = direction / direction.norm()
        directions[layer] = direction
    return directions, raw_norms


def forward(model, tokenizer, source_texts, batch_size=8, hidden_layers=-1, prompt_only=False, pooling='mean'):
    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]
    source_embeddings = {layer: [] for layer in hidden_layers}
    if prompt_only:
        pos_texts = [item[0] for item in source_texts]
        neg_texts = [item[2] for item in source_texts]
    else:
        pos_texts = [item[1] for item in source_texts]
        neg_texts = [item[3] for item in source_texts]

    all_texts = []
    for p, n in zip(pos_texts, neg_texts):
        all_texts.append(p)
        all_texts.append(n)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i: i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model(**inputs, output_hidden_states=True)
                batch_hidden_states = get_hidden_states(outputs, inputs['attention_mask'], hidden_layers, pooling=pooling)
                del outputs
                for layer, states in batch_hidden_states.items():
                    source_embeddings[layer].append(states)
                del batch_hidden_states
            del inputs
            torch.cuda.empty_cache()

    final_results = {}
    for layer in hidden_layers:
        layer_data = torch.cat(source_embeddings[layer], dim=0)
        final_results[layer] = layer_data[::2] - layer_data[1::2]
    return final_results


def get_direction(model, tokenizer, source_texts, rep_token=-1, hidden_layers=-1, batch_size=8, use_pca=True, prompt_only=True, pooling='mean'):
    if not isinstance(hidden_layers, list):
        hidden_layers = [hidden_layers]
    hidden_states = forward(model, tokenizer, source_texts, batch_size=batch_size, hidden_layers=hidden_layers, prompt_only=prompt_only, pooling=pooling)
    directions, raw_norms = get_rep_directions(hidden_states, hidden_layers, use_pca=use_pca)
    return directions, raw_norms


def merge_directions(*directions):
    if not directions:
        return {}
    common_keys = directions[0].keys()
    merged = {}
    for layer in common_keys:
        layer_tensors = [d[layer] for d in directions]
        stacked = torch.stack(layer_tensors, dim=0)
        flat = stacked.squeeze(1)
        merged_vec = torch.mean(flat, dim=0)
        merged[layer] = merged_vec.unsqueeze(0)
    return merged


def _load_or_compute_forward_direction(model, tokenizer, fwd_transition, all_pairs, hidden_layers, cache_dir='outputs/cefr', use_pca=True):
    cache_path = os.path.join(cache_dir, f'{fwd_transition}_cached_directions.pt')
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if isinstance(cached, tuple):
            return cached[0]
        return cached
    if fwd_transition not in all_pairs or len(all_pairs[fwd_transition]) < 2:
        return None
    dirs, raw_norms = get_direction(
        model, tokenizer, all_pairs[fwd_transition],
        -1, hidden_layers, use_pca=use_pca,
    )
    torch.save((dirs, raw_norms), cache_path)
    return dirs


def get_competitor_directions(model, tokenizer, transition, all_pairs, hidden_layers, cache_dir='outputs/cefr', use_pca=True):
    src, tgt = transition.split('_to_')
    src_idx = CEFR_TO_ID[src]
    tgt_idx = CEFR_TO_ID[tgt]
    competitors = []
    for z in CEFR_ORDER:
        z_idx = CEFR_TO_ID[z]
        if z_idx >= src_idx or z_idx == tgt_idx:
            continue
        fwd = f'{z}_to_{src}'
        d = _load_or_compute_forward_direction(model, tokenizer, fwd, all_pairs, hidden_layers, cache_dir, use_pca)
        if d is not None:
            print(f"  competitor: {fwd}")
            competitors.append(d)
    return competitors


def project_out_competitor_subspace(direction, competitor_dirs_list, layer_ids):
    projected = {}
    for layer in layer_ids:
        d = direction[layer][0].float()
        d_orig_norm = d.norm()
        cols = []
        for comp_dirs in competitor_dirs_list:
            if layer not in comp_dirs:
                continue
            c = comp_dirs[layer][0].float()
            if c.norm() < 1e-8:
                continue
            cols.append(c)
        if not cols:
            projected[layer] = direction[layer]
            continue
        C = torch.stack(cols, dim=1)
        Q, _ = torch.linalg.qr(C, mode='reduced')
        d_resid = d - Q @ (Q.T @ d)
        retained = d_resid.norm() / (d_orig_norm + 1e-8)
        if retained < 0.1:
            d_resid = d
            print(f"  Layer {layer}: competitor projection too aggressive (retained {retained:.2f}), keeping original")
        else:
            print(f"  Layer {layer}: retained {retained:.2f} after projecting out {len(cols)} competitors")
        d_resid = d_resid / (d_resid.norm() + 1e-8)
        projected[layer] = d_resid.unsqueeze(0)
    return projected


CEFR_DESCRIPTORS = {
    "A1": "A1 beginner level: very simple vocabulary, short sentences, direct wording, concrete meaning, minimal grammar complexity.",
    "A2": "A2 elementary level: simple everyday vocabulary, mostly short sentences, clear sequencing, limited but slightly richer grammar.",
    "B1": "B1 intermediate level: clear standard language, familiar vocabulary, some connected clauses, modest detail, natural but not highly complex phrasing.",
    "B2": "B2 upper-intermediate level: more fluent and detailed phrasing, broader vocabulary, clear argumentation, varied sentence structures, but still accessible.",
    "C1": "C1 advanced level: fluent, precise, idiomatic wording, nuanced vocabulary, complex sentence structures, natural cohesion, and sophisticated expression.",
    "C2": "C2 proficient level: highly precise, subtle, flexible, and polished language, with sophisticated syntax and near-native control of tone and nuance.",
}


CEFR_ASPECTS = ['vocabulary', 'syntax', 'cohesion', 'fluency']

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))


# English source data used to extract steering directions (fallback if no cache).
with open('outputs/cefr/cross_cefr_similar_prompts.json', 'r') as f:
    target_docs_responses = json.load(f)

# Cross-lingual evaluation inputs.
with open('data/cefr/cefr_de_nl_texts.json', 'r') as f:
    xlingual_inputs = json.load(f)


cefr_classifier = CEFRClassifier(
    model_name_or_path=cefr_classifier_name,
    device='cuda',
    max_length=512,
)


def build_directions_for_transition(transition: str, layer_ids: List[int]):
    """Compute (or load) the English-derived steering direction for a transition."""
    src, tgt = transition.split('_to_')
    is_reverse = CEFR_TO_ID[src] > CEFR_TO_ID[tgt]
    direction_transition = f'{tgt}_to_{src}' if is_reverse else transition
    sign = -1 if is_reverse else 1

    aspect_directions = []
    for aspect in CEFR_ASPECTS:
        aspect_cache = f'outputs/cefr/tgt_caches_{aspect}_{direction_transition}.json'
        if os.path.exists(aspect_cache):
            with open(aspect_cache) as f:
                aspect_pairs = json.load(f)[direction_transition]
            print(f"  Loading {aspect} aspect: {len(aspect_pairs)} pairs (from {direction_transition})")
            d, _ = get_direction(model, tokenizer, aspect_pairs, rep_token, hidden_layers, use_pca=True)
            aspect_directions.append(d)

    if aspect_directions:
        directions = merge_directions(*aspect_directions)
        raw_norms = {layer: float(directions[layer].norm().item()) for layer in directions}
    else:
        cache_path = f'outputs/cefr/{direction_transition}_cached_directions.pt'
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu')
            if isinstance(cached, tuple):
                directions, raw_norms = cached
            else:
                directions = cached
                raw_norms = {layer: float(cached[layer].norm().item()) for layer in cached}
        else:
            directions, raw_norms = get_direction(
                model, tokenizer, target_docs_responses[direction_transition],
                rep_token, hidden_layers, use_pca=True,
            )
            torch.save((directions, raw_norms), cache_path)
            print(f"Saved fallback directions for {direction_transition}")

    if is_reverse:
        competitor_dirs = get_competitor_directions(
            model, tokenizer, transition, target_docs_responses, hidden_layers,
        )
        if competitor_dirs:
            print(f"Projecting out {len(competitor_dirs)} competitor directions for {transition}")
            directions = project_out_competitor_subspace(directions, competitor_dirs, layer_ids)
        else:
            print(f"No competitor directions available for {transition}, skipping projection")

    return directions, raw_norms, sign


# Reverse C1->X transitions: borrow alpha schedule that worked on English.
ALPHA_BY_TRANSITION = {
    'B2_to_C1': 0.7, 'B1_to_C1': 2.0, 'A1_to_C1': 1.0,
    'C1_to_B2': 0.85, 'C1_to_B1': 1.2, 'C1_to_A2': 0.55, 'C1_to_A1': 1.0,
}

# (source_lang, transition) pairs to evaluate. Filter to languages that
# actually have samples at the source level in the xlingual data.
TRANSITIONS = ['B1_to_C1']
# LANGS = ['de', 'it']
LANGS = ['nl']
N_INPUTS_PER_LANG = 20

layer_ids = [-27, -26, -25, -24, -18, -17, -16, -15, -14, -11]


for transition in TRANSITIONS:
    src, tgt = transition.split('_to_')
    print(f"\n############ Building English direction for {transition} ############")
    directions, raw_norms, sign = build_directions_for_transition(transition, layer_ids)

    max_norm = max(raw_norms.get(l, 1.0) for l in layer_ids) + 1e-8
    layer_weights = {l: raw_norms.get(l, 1.0) / max_norm for l in layer_ids}
    alpha = ALPHA_BY_TRANSITION.get(transition, 0.4)

    activations = {}
    for layer in layer_ids:
        activations[layer] = torch.tensor(
            alpha * layer_weights[layer] * directions[layer][0] * sign
        ).to(model.device)

    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_ids, block_name="decoder_block")
    wrapped_model.set_controller(layer_ids, activations, masks=1, normalize=False)

    for lang in LANGS:
        lang_pool = xlingual_inputs.get(src, {}).get(lang, [])
        if not lang_pool:
            print(f"  [skip] no {src} samples for lang={lang}")
            continue
        all_data = lang_pool[:N_INPUTS_PER_LANG]
        lang_name = LANG_NAMES.get(lang, lang)

        out_path = f'outputs/cefr/xlingual_{lang}_{transition}_v0.jsonl'
        results = {}
        start_time = time.time()
        print(f"\n=== {transition}  lang={lang} ({lang_name})  alpha={alpha}  n={len(all_data)} ===")

        with open(out_path, 'w') as out_f:
            for idx, inp in enumerate(all_data):
                if not isinstance(inp, str) or len(inp) < 10:
                    continue

                user_prompt = (
                    f"Rewrite the {lang_name} text below at CEFR {tgt} proficiency. "
                    f"{CEFR_DESCRIPTORS[tgt]}\n"
                    f"Text: {inp}"
                )
                input_messages = [
                    {"role": "system", "content": f"Give ONLY an output. No explanation."},
                    {"role": "user", "content": user_prompt},
                ]
                source_input_prompt = tokenizer.apply_chat_template(
                    input_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                encoded_inputs = tokenizer([source_input_prompt], return_tensors='pt')

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=True):
                        outputs = wrapped_model.generate(
                            **encoded_inputs.to(model.device),
                            max_new_tokens=256,
                            do_sample=False,
                            use_cache=True,
                        ).detach().cpu()
                        sanity_generation = tokenizer.decode(outputs[0]).replace(source_input_prompt, "").lstrip('<|begin_of_text|>').rstrip('<|eot_id|>').strip()
                        if 'assistant<|end_header_id|>' in sanity_generation:
                            sanity_generation = sanity_generation[sanity_generation.find('assistant<|end_header_id|>'):].replace('assistant<|end_header_id|>', '').strip()

                output_cefr = cefr_classifier.classify_one(sanity_generation)
                output_pred = output_cefr["label"]
                results[output_pred] = results.get(output_pred, 0) + 1

                print(f"++INPUT [{lang} {src}]:", inp[:200])
                print("++OUTPUT:", sanity_generation[:300])
                print(f"++OUTPUT pred [{tgt}?]:", output_pred, output_cefr["confidence"])
                print("++TARGET MATCH:", output_pred == tgt)
                print('===============')

                json.dump({
                    'id': idx,
                    'lang': lang,
                    'transition': transition,
                    'input': inp,
                    'output': sanity_generation,
                    'predicted_cefr': output_pred,
                    'confidence': output_cefr["confidence"],
                    'alpha': float(alpha),
                }, out_f, ensure_ascii=False)
                out_f.write('\n')

        print(f"=== {transition} {lang} results (alpha={alpha}) ===")
        print(results)
        print('END', time.time() - start_time)

    wrapped_model.reset()
    wrapped_model.unwrap()
