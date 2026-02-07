# pointllm/model/grasp_point_tokenizer.py
import torch
import torch.nn as nn

class GRASPPointTokenizer(nn.Module):
    """
    points -> GRASP encoder -> adapter -> GRASP decoder -> adapter -> [B, P, D_llm]
    """
    def __init__(
        self,
        grasp_encoder: nn.Module,
        grasp_decoder: nn.Module,
        point_token_len: int,
        llm_hidden_size: int,
        cg_enc: int = None,
        cg_dec: int = None,
        use_mlp: bool = False,
    ):
        super().__init__()
        self.grasp_encoder = grasp_encoder
        self.grasp_decoder = grasp_decoder
        self.point_token_len = point_token_len
        self.llm_hidden_size = llm_hidden_size

        # enc_adapter：如果 encoder 输出维度 != decoder 需要维度，你就用 Linear 对齐
        if cg_enc is not None and cg_dec is not None and cg_enc != cg_dec:
            self.enc_adapter = nn.Sequential(nn.LayerNorm(cg_enc), nn.Linear(cg_enc, cg_dec))
            in_dec = cg_dec
        else:
            self.enc_adapter = nn.Identity()
            in_dec = cg_dec  # 允许为 None（后面会用 decoder 输出推断）

        # dec_adapter：decoder feature -> LLM hidden
        # 这里建议 LayerNorm + Linear，够稳
        self.dec_adapter = None
        self._dec_adapter_in = in_dec
        self._use_mlp = use_mlp

    def _ensure_dec_adapter(self, feat_dim: int):
        if self.dec_adapter is not None:
            return
        if self._use_mlp:
            self.dec_adapter = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, self.llm_hidden_size),
                nn.GELU(),
                nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
            )
        else:
            self.dec_adapter = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, self.llm_hidden_size),
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: [B, N, C]
        return: point_embeds [B, P, D_llm]
        """
        z = self.grasp_encoder(points)
        if isinstance(z, dict):
            z = z.get("x") or z.get("features") or z.get("latent")
        assert z is not None, "GRASP encoder RunsenXu_graspnet_enc_dec_r04 missing (dict key mismatch)."

        z = self.enc_adapter(z)

        h = self.grasp_decoder(z)
        if isinstance(h, dict):
            h = h.get("x") or h.get("features") or h.get("decoded_feat")
        assert h is not None, "GRASP decoder RunsenXu_graspnet_enc_dec_r04 missing (dict key mismatch)."

        # h: [B, P_dec, Cg]
        if h.size(1) >= self.point_token_len:
            h = h[:, : self.point_token_len]
        else:
            pad = self.point_token_len - h.size(1)
            h = torch.cat([h, h[:, -1:].repeat(1, pad, 1)], dim=1)

        self._ensure_dec_adapter(h.size(-1))
        point_embeds = self.dec_adapter(h)  # [B, P, D]
        return point_embeds
