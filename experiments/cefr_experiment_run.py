from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import time
import torch
from typing import List, Any, Dict, Union
import numpy as np
from control_block import WrappedReadingVecModel


model_name = 'unsloth/Llama-3.2-3B-Instruct'
# model_name = 'Qwen/Qwen3-4B'

cefr_classifier_name = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_ID = {level: i for i, level in enumerate(CEFR_ORDER)}


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
model.eval()



class CEFRClassifier:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        max_length: int = 512,
    ):
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(self.device)

        self.model.eval()

        # Normalize id2label because HF configs sometimes load keys as strings.
        raw_id2label = getattr(self.model.config, "id2label", None)

        if raw_id2label:
            self.id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            self.id2label = {
                0: "A1",
                1: "A2",
                2: "B1",
                3: "B2",
                4: "C1",
                5: "C2",
            }

    @torch.inference_mode()
    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        results = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            pred_ids = torch.argmax(probs, dim=-1)

            for i, pred_id in enumerate(pred_ids):
                pred_id_int = int(pred_id.item())
                pred_label = self.id2label[pred_id_int]
                confidence = float(probs[i, pred_id_int].item())

                prob_dict = {
                    self.id2label[j]: float(probs[i, j].item())
                    for j in range(probs.shape[-1])
                }

                results.append({
                    "label": pred_label,
                    "confidence": confidence,
                    "probs": prob_dict,
                })

        return results

    def classify_one(self, text: str) -> Dict[str, Any]:
        return self.classify_batch([text], batch_size=1)[0]


def masked_mean(hidden_states, mask):
    """
    Computes the mean of hidden states ignoring padding tokens.
    hidden_states: (Batch, Seq_Len, Hidden_Dim)
    mask: (Batch, Seq_Len)
    """
    # Expand mask to match hidden states shape: (B, S, 1)
    mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
    
    # Sum the valid tokens
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    
    # Count the number of valid tokens (avoid div by zero)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    
    return sum_embeddings / sum_mask


def last_token_pool(hidden_states, mask):
    """Extract hidden state at the last non-padding token for each sequence."""
    last_idx = mask.sum(dim=1) - 1  # (B,)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]


def get_hidden_states(
    outputs,
    attention_mask,
    hidden_layers: Union[List[int], int]=-1,
    pooling: str = 'mean',
):
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


def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x,axis=0,keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x - mean


def get_rep_directions(hidden_states, hidden_layers, n_components: int = 1, use_pca: bool = True, normalize: bool = True):
    """Get direction vectors for each layer.

    When use_pca=True, computes the top singular vector of the raw (uncentered)
    difference matrix — this finds the strongest axis of systematic variation
    rather than averaging out noise from near-identical pairs.

    When normalize=True, L2-normalizes each direction to unit length so that
    alpha has consistent meaning across transitions with different raw norms.
    """
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


def forward(model, tokenizer, source_texts, batch_size: int = 8, hidden_layers: Union[List[int], int]=-1, prompt_only: bool = False, pooling: str = 'mean'):
    """
    Optimized forward pass processing texts in batches.

    When prompt_only=True, uses just the instruction prompts (item[0], item[2])
    instead of the full prompt+completion (item[1], item[3]).  This extracts how
    the model represents "rewrite at level X" without noise from near-identical
    completion texts — critical for adjacent CEFR levels.

    pooling: 'last_token' extracts the last non-padding token's hidden state
             (sharper signal at the generation decision point).
             'mean' uses masked mean over all tokens.
    """
    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]

    source_embeddings = {layer: [] for layer in hidden_layers}

    if prompt_only:
        pos_texts = [item[0] for item in source_texts]
        neg_texts = [item[2] for item in source_texts]
    else:
        pos_texts = [item[1] for item in source_texts]
        neg_texts = [item[3] for item in source_texts]
    
    # Combine into one list to maximize batch usage [pos1, neg1, pos2, neg2, ...]
    # This keeps the math (pos - neg) easy later
    all_texts = []
    for p, n in zip(pos_texts, neg_texts):
        all_texts.append(p)
        all_texts.append(n)
    
    model.eval() # Ensure model is in eval mode
    
    with torch.no_grad():
        # Iterate in chunks (batches)
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            
            # Tokenize entire batch at once
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to('cuda')
            
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract states using the attention mask to handle padding
                batch_hidden_states = get_hidden_states(
                    outputs,
                    inputs['attention_mask'],
                    hidden_layers,
                    pooling=pooling,
                )
                del outputs
                for layer, states in batch_hidden_states.items():
                    source_embeddings[layer].append(states)
                del batch_hidden_states
            del inputs
            torch.cuda.empty_cache()

    # Consolidate and compute directions
    final_results = {}
    for layer in hidden_layers:
        # Concatenate all batches: Shape (Total_Samples, Hidden_Dim)
        layer_data = torch.cat(source_embeddings[layer], dim=0)
        
        # Compute difference: [::2] are positives, [1::2] are negatives
        # This performs the subtraction (Pos - Neg) purely on GPU
        final_results[layer] = layer_data[::2] - layer_data[1::2]
        
    return final_results


def get_overshoot_direction(
    model, tokenizer, transition, all_pairs, hidden_layers,
    cache_dir='outputs/cefr', use_pca=True,
):
    """For transition X->Y, compute direction Y->Z (next CEFR level up).

    This captures the 'overshoot' axis that we want to project out.
    Returns (directions, raw_norms) or (None, None) if unavailable.
    """
    src, tgt = transition.split('_to_')
    tgt_idx = CEFR_TO_ID.get(tgt, -1)
    if tgt_idx < 0 or tgt_idx >= len(CEFR_ORDER) - 1:
        return None, None

    overshoot_level = CEFR_ORDER[tgt_idx + 1]
    overshoot_transition = f'{tgt}_to_{overshoot_level}'

    cache_path = os.path.join(cache_dir, f'{overshoot_transition}_cached_directions.pt')
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        if isinstance(cached, tuple):
            return cached
        raw_norms = {layer: float(cached[layer].norm().item()) for layer in cached}
        return cached, raw_norms

    if overshoot_transition not in all_pairs or len(all_pairs[overshoot_transition]) < 2:
        return None, None

    dirs, raw_norms = get_direction(
        model, tokenizer, all_pairs[overshoot_transition],
        -1, hidden_layers, use_pca=use_pca,
    )
    torch.save((dirs, raw_norms), cache_path)
    return dirs, raw_norms


def _load_or_compute_forward_direction(model, tokenizer, fwd_transition, all_pairs, hidden_layers, cache_dir='outputs/cefr', use_pca=True):
    """Load cached forward direction (Y_to_C1 style) or compute and cache it."""
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


def get_reverse_overshoot_direction(model, tokenizer, transition, all_pairs, hidden_layers, cache_dir='outputs/cefr', use_pca=True):
    """For reverse transition C1->Y, build overshoot axis Y->W (W = one level below Y).

    Constructs d(Y_to_W) ~= d(Y_to_C1) - d(W_to_C1) from cached forward
    directions. Returns None if Y is the bottom level (no overshoot possible)
    or if a needed forward direction can't be obtained.
    """
    src, tgt = transition.split('_to_')
    tgt_idx = CEFR_TO_ID.get(tgt, -1)
    if tgt_idx <= 0:
        return None
    below_level = CEFR_ORDER[tgt_idx - 1]

    fwd_tgt = f'{tgt}_to_{src}'
    fwd_below = f'{below_level}_to_{src}'

    d_tgt = _load_or_compute_forward_direction(model, tokenizer, fwd_tgt, all_pairs, hidden_layers, cache_dir, use_pca)
    d_below = _load_or_compute_forward_direction(model, tokenizer, fwd_below, all_pairs, hidden_layers, cache_dir, use_pca)
    if d_tgt is None or d_below is None:
        return None

    overshoot = {}
    for layer in d_tgt:
        if layer not in d_below:
            continue
        diff = d_tgt[layer][0].float() - d_below[layer][0].float()
        overshoot[layer] = diff.unsqueeze(0)
    return overshoot


def get_competitor_directions(model, tokenizer, transition, all_pairs, hidden_layers, cache_dir='outputs/cefr', use_pca=True):
    """For reverse transition src->tgt, return list of forward directions
    d(Z_to_src) for every CEFR level Z strictly below src that is not the
    target. These are the 'competing target' directions whose overlap with
    the steering direction we want to remove.
    """
    src, tgt = transition.split('_to_')
    src_idx = CEFR_TO_ID[src]
    tgt_idx = CEFR_TO_ID[tgt]
    competitors = []
    for z in CEFR_ORDER:
        z_idx = CEFR_TO_ID[z]
        if z_idx >= src_idx or z_idx == tgt_idx:
            continue
        fwd = f'{z}_to_{src}'
        d = _load_or_compute_forward_direction(
            model, tokenizer, fwd, all_pairs, hidden_layers, cache_dir, use_pca,
        )
        if d is not None:
            print(f"  competitor: {fwd}")
            competitors.append(d)
    return competitors


def project_out_competitor_subspace(direction, competitor_dirs_list, layer_ids):
    """Project `direction` orthogonal to the subspace spanned by competitors.

    Uses QR on the stacked competitor matrix to build an orthonormal basis Q
    of the competitor subspace, then returns d - Q Q^T d. This is the true
    orthogonal complement (iterative Gram-Schmidt against non-orthogonal
    competitors leaves residual components in earlier-projected vectors).
    Falls back to the original direction per-layer if too little norm
    survives projection.
    """
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

        C = torch.stack(cols, dim=1)  # [hidden_dim, k]
        Q, _ = torch.linalg.qr(C, mode='reduced')  # [hidden_dim, k_eff]
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


def project_out_overshoot(direction, overshoot_dir, layer_ids):
    """Per-layer Gram-Schmidt: remove overshoot component, re-normalize."""
    projected = {}
    for layer in layer_ids:
        d = direction[layer][0].float()
        ov = overshoot_dir[layer][0].float()
        ov_unit = ov / (ov.norm() + 1e-8)
        d_proj = d - (d @ ov_unit) * ov_unit

        retained = d_proj.norm() / (d.norm() + 1e-8)
        if retained < 0.1:
            d_proj = d
            print(f"  Layer {layer}: projection too aggressive (retained {retained:.2f}), keeping original")
        else:
            print(f"  Layer {layer}: retained {retained:.2f} of direction norm after projection")

        d_proj = d_proj / (d_proj.norm() + 1e-8)
        projected[layer] = d_proj.unsqueeze(0)
    return projected


def get_direction(model, tokenizer, source_texts, rep_token: int = -1, hidden_layers: Union[List[int], int]=-1, batch_size: int = 8, use_pca: bool = True, prompt_only: bool = True, pooling: str = 'mean'):
    if not isinstance(hidden_layers, list):
        assert isinstance(hidden_layers, int)
        hidden_layers = [hidden_layers]

    start_time = time.time()
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


CEFR_ASPECTS = ['vocabulary', 'syntax', 'cohesion', 'fluency']


rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))


target_docs_responses = {}
with open(f'outputs/cefr/cross_cefr_similar_prompts.json', 'r') as f:
    target_docs_responses = json.load(f)


cefr_classifier = CEFRClassifier(
    model_name_or_path=cefr_classifier_name,
    device='cuda',
    max_length=512,
)

CEFR_DESCRIPTORS = {
    "A1": (
        "A1 beginner English: very simple vocabulary, short sentences, "
        "direct wording, concrete meaning, minimal grammar complexity."
    ),
    "A2": (
        "A2 elementary English: simple everyday vocabulary, mostly short sentences, "
        "clear sequencing, limited but slightly richer grammar."
    ),
    "B1": (
        "B1 intermediate English: clear standard language, familiar vocabulary, "
        "some connected clauses, modest detail, natural but not highly complex phrasing."
    ),
    "B2": (
        "B2 upper-intermediate English: more fluent and detailed phrasing, broader vocabulary, "
        "clear argumentation, varied sentence structures, but still accessible."
    ),
    "C1": (
        "C1 advanced English: fluent, precise, idiomatic wording, nuanced vocabulary, "
        "complex sentence structures, natural cohesion, and sophisticated expression."
    ),
    "C2": (
        "C2 proficient English: highly precise, subtle, flexible, and polished language, "
        "with sophisticated syntax and near-native control of tone and nuance."
    ),
}


for transition in ['C1_to_B1']:
    all_data = []
    src, tgt = transition.split('_to_')

    # Reverse transition: src is a higher CEFR level than tgt.
    # We reuse the cached forward direction (tgt_to_src) and flip its sign.
    is_reverse = CEFR_TO_ID[src] > CEFR_TO_ID[tgt]
    direction_transition = f'{tgt}_to_{src}' if is_reverse else transition
    sign = -1 if is_reverse else 1

    with open('data/cefr/cefr_samples.json', 'r') as f:
        all_data = json.load(f)[src][:20]

    start_time = time.time()
    results = {}
    with open(f'outputs/cefr/activation_wrapped_texts_{transition}_v0.jsonl', 'w') as out_f:
        layer_ids = [-27, -26, -25, -24, -18, -17, -16, -15, -14, -11]

        # Multi-aspect direction computation. For reverse transitions we load
        # the forward aspect caches (tgt_to_src) and rely on `sign` to flip.
        aspect_directions = []
        for aspect in CEFR_ASPECTS:
            aspect_cache = f'outputs/cefr/tgt_caches_{aspect}_{direction_transition}.json'
            if os.path.exists(aspect_cache):
                with open(aspect_cache) as f:
                    aspect_pairs = json.load(f)[direction_transition]
                print(f"  Loading {aspect} aspect: {len(aspect_pairs)} pairs (from {direction_transition})")
                d, _ = get_direction(
                    model, tokenizer, aspect_pairs,
                    rep_token, hidden_layers, use_pca=True,
                )
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

        # Orthogonal projection: remove overshoot component. For forward
        # transitions overshoot is the next level up. For reverse transitions
        # it's the next level down, constructed via subtraction of cached
        # forward directions.
        if is_reverse:
            # Residualize against ALL other candidate target levels, not just
            # the next-level overshoot. For C1->B1 this strips the components
            # of the steering direction that are shared with d(B2->C1),
            # d(A2->C1), d(A1->C1), leaving the part that is specifically B1.
            competitor_dirs = get_competitor_directions(
                model, tokenizer, transition, target_docs_responses, hidden_layers,
            )
            if competitor_dirs:
                print(f"Projecting out {len(competitor_dirs)} competitor directions for {transition}")
                directions = project_out_competitor_subspace(directions, competitor_dirs, layer_ids)
            else:
                print(f"No competitor directions available for {transition}, skipping projection")
        else:
            overshoot_dirs, _ = get_overshoot_direction(
                model, tokenizer, transition, target_docs_responses, hidden_layers,
            )
            if overshoot_dirs is not None:
                print(f"Projecting out overshoot for {transition}")
                directions = project_out_overshoot(directions, overshoot_dirs, layer_ids)
            else:
                print(f"No overshoot direction for {transition} (target is C2), skipping projection")

        # Layer-weighted alpha: scale each layer by its raw direction norm
        max_norm = max(raw_norms.get(l, 1.0) for l in layer_ids) + 1e-8
        layer_weights = {l: raw_norms.get(l, 1.0) / max_norm for l in layer_ids}

        ALPHA_BY_TRANSITION = {
            'B2_to_C1': 0.7,
            'B1_to_C1': 0.6,
            'A1_to_C1': 1.0,
            'C1_to_B2': 0.85,
            'C1_to_B1': 1.2,
            'C1_to_A2': 0.55,
            'C1_to_A1': 1.0,
        }
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

        for idx, inp in enumerate(all_data):
            if len(inp) < 10:
                continue

            user_prompt = (
                f"Rewrite the text below at CEFR {tgt} proficiency. "
                f"{CEFR_DESCRIPTORS[tgt]}\n"
                f"Text: {inp}"
            )
            input_messages = [
                {"role": "system", "content": "Give an output ONLY. No explanation."},
                {"role": "user", "content": user_prompt}
            ]
            source_input_prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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
            if output_pred not in results:
                results[output_pred] = 0
            results[output_pred] += 1
            print(f"++INPUT in {src}:", inp)
            print("++OUTPUT:", sanity_generation)
            print(f"++OUTPUT in {tgt}:", output_cefr["label"], output_cefr["confidence"])
            print("++TARGET MATCH:", output_pred == tgt)
            print('===============')

            json.dump({
                'id': idx,
                'transition': transition,
                'input': inp,
                'output': sanity_generation,
                'predicted_cefr': output_pred,
                'confidence': output_cefr["confidence"],
                'alpha': float(alpha),
            }, out_f, ensure_ascii=False)
            out_f.write('\n')

        wrapped_model.reset()
        wrapped_model.unwrap()
    print(f"\n=== {transition} results (alpha={alpha}) ===")
    print(results)
    print('END', time.time() - start_time)
