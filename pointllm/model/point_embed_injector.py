# pointllm/model/point_embed_injector.py
import torch

def inject_point_embeds(model, input_ids, point_embeds, point_patch_token_id):
    """
    model: LLM model (HF)
    input_ids: [B, L]
    point_embeds: [B, P, D]
    return: inputs_embeds [B, L, D]
    """
    inputs_embeds = model.get_input_embeddings()(input_ids)  # [B, L, D]
    mask = (input_ids == point_patch_token_id)               # [B, L]
    P = point_embeds.size(1)

    # 每个样本必须至少有 P 个占位 token
    if mask.sum(dim=1).min().item() < P:
        raise ValueError("Not enough <point_patch> tokens in prompt for some samples.")

    B = input_ids.size(0)
    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)[:P]
        inputs_embeds[b, idx] = point_embeds[b]
    return inputs_embeds
