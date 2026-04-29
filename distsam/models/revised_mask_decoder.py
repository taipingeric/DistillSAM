import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim] + hidden, hidden + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class RevisedMaskDecoder(nn.Module):
    """Frozen SAM mask decoder plus a trainable class head for semantic segmentation."""

    def __init__(
        self,
        sam_mask_decoder: nn.Module,
        num_classes: int,
        freeze_sam_decoder: bool = True,
        class_head_hidden_dim: int = 256,
        use_default_mask_tokens: bool = True,
        extra_mask_tokens: int = 0,
    ) -> None:
        super().__init__()
        if not use_default_mask_tokens:
            raise NotImplementedError("DistillSAM currently requires use_default_mask_tokens=True.")
        if extra_mask_tokens != 0:
            raise NotImplementedError("DistillSAM currently does not implement extra_mask_tokens.")

        self.sam_mask_decoder = sam_mask_decoder
        self.num_classes = int(num_classes)
        self.use_default_mask_tokens = bool(use_default_mask_tokens)
        self.extra_mask_tokens = int(extra_mask_tokens)
        self.debug_shapes = True
        self._printed_shapes = False

        if freeze_sam_decoder:
            self.sam_mask_decoder.eval()
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad_(False)

        transformer_dim = self.sam_mask_decoder.transformer_dim
        self.class_head = MLP(
            input_dim=transformer_dim,
            hidden_dim=class_head_hidden_dim,
            output_dim=self.num_classes,
            num_layers=3,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        return_tokens: bool = False,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        mask_logits_list = []
        class_logits_list = []
        iou_predictions_list = []
        mask_tokens_out_list = []
        final_attn_list = []

        for index in range(image_embeddings.shape[0]):
            mask_logits, class_logits, iou_predictions, mask_tokens_out, attention_maps = self._predict_one(
                image_embeddings=image_embeddings[index : index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings[index : index + 1],
                dense_prompt_embeddings=dense_prompt_embeddings[index : index + 1],
                return_attn=return_attn,
            )
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)
            iou_predictions_list.append(iou_predictions)
            mask_tokens_out_list.append(mask_tokens_out)
            if return_attn:
                final_attn_list.append(attention_maps["final_attn_token_to_image"])

        mask_logits_low_res = torch.cat(mask_logits_list, dim=0)
        class_logits = torch.cat(class_logits_list, dim=0)
        iou_predictions = torch.cat(iou_predictions_list, dim=0)
        mask_tokens_out = torch.cat(mask_tokens_out_list, dim=0)
        semantic_logits_low_res = torch.einsum(
            "bmhw,bmk->bkhw",
            mask_logits_low_res,
            class_logits,
        )

        if self.debug_shapes and not self._printed_shapes:
            print(f"[shape] mask_tokens_out: {tuple(mask_tokens_out.shape)}")
            print(f"[shape] mask_logits_low_res: {tuple(mask_logits_low_res.shape)}")
            print(f"[shape] class_logits: {tuple(class_logits.shape)}")
            print(f"[shape] semantic_logits_low_res: {tuple(semantic_logits_low_res.shape)}")
            self._printed_shapes = True

        output = {
            "semantic_logits_low_res": semantic_logits_low_res,
            "mask_logits_low_res": mask_logits_low_res,
            "class_logits": class_logits,
            "iou_predictions": iou_predictions,
        }
        if return_tokens:
            output["mask_tokens_out"] = mask_tokens_out
        if return_attn:
            output["attention_maps"] = {
                "final_attn_token_to_image": torch.cat(final_attn_list, dim=0),
            }
        return output

    def _predict_one(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        return_attn: bool = False,
    ):
        decoder = self.sam_mask_decoder
        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        batch_size, channels, height, width = src.shape

        attention_maps = {}
        if return_attn:
            hs, src, final_attn = self._transformer_forward_with_final_attn(
                decoder.transformer,
                src,
                pos_src,
                tokens,
            )
            attention_maps["final_attn_token_to_image"] = final_attn
        else:
            hs, src = decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + decoder.num_mask_tokens), :]

        src = src.transpose(1, 2).view(batch_size, channels, height, width)
        upscaled_embedding = decoder.output_upscaling(src)
        hyper_in_list = []
        for token_index in range(decoder.num_mask_tokens):
            hyper_in_list.append(
                decoder.output_hypernetworks_mlps[token_index](mask_tokens_out[:, token_index, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        batch_size, channels, height, width = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(batch_size, channels, height * width)).view(
            batch_size,
            -1,
            height,
            width,
        )
        iou_predictions = decoder.iou_prediction_head(iou_token_out)
        class_logits = self.class_head(mask_tokens_out)
        return masks, class_logits, iou_predictions, mask_tokens_out, attention_maps

    @staticmethod
    def _attention_forward_with_probs(attention_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = attention_module.q_proj(q)
        k = attention_module.k_proj(k)
        v = attention_module.v_proj(v)

        q = attention_module._separate_heads(q, attention_module.num_heads)
        k = attention_module._separate_heads(k, attention_module.num_heads)
        v = attention_module._separate_heads(v, attention_module.num_heads)

        _, _, _, channels_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(channels_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = attention_module._recombine_heads(out)
        out = attention_module.out_proj(out)
        return out, attn

    def _transformer_forward_with_final_attn(
        self,
        transformer,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ):
        batch_size, channels, height, width = image_embedding.shape
        del batch_size, channels, height, width
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding
        for layer in transformer.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out, final_attn = self._attention_forward_with_probs(
            transformer.final_attn_token_to_image,
            q=q,
            k=k,
            v=keys,
        )
        queries = queries + attn_out
        queries = transformer.norm_final_attn(queries)
        return queries, keys, final_attn
