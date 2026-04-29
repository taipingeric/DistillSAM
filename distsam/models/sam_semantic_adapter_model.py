import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from .revised_mask_decoder import RevisedMaskDecoder


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ANYTHING_ROOT = REPO_ROOT / "segment-anything"
if str(SEGMENT_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGMENT_ANYTHING_ROOT))

from segment_anything import sam_model_registry  # noqa: E402


class SamSemanticAdapterModel(nn.Module):
    """DistillSAM adapter model for semantic or binary foreground segmentation."""

    def __init__(
        self,
        sam_checkpoint: str,
        num_classes: int,
        segmentation_mode: str = "semantic",
        use_revised_decoder: bool = True,
        adapter_type: str = "pvt_cross",
        pvt_pretrained: str = "",
        adapter_layers: List[int] = None,
        token_stride: int = 1,
        num_prompt_tokens: int = 16,
        freeze_sam: bool = True,
        class_head_hidden_dim: int = 256,
        use_default_mask_tokens: bool = True,
        extra_mask_tokens: int = 0,
        debug_shapes: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.segmentation_mode = segmentation_mode
        self.use_revised_decoder = bool(use_revised_decoder)
        self.adapter_type = adapter_type
        self.debug_shapes = bool(debug_shapes)
        self._printed_shapes = False

        if self.segmentation_mode == "auto":
            self.segmentation_mode = "binary" if self.num_classes <= 2 else "semantic"
        if self.segmentation_mode not in {"semantic", "binary"}:
            raise ValueError(f"Unsupported segmentation_mode: {self.segmentation_mode}")
        if self.segmentation_mode == "semantic" and not self.use_revised_decoder:
            raise ValueError("DistillSAM semantic training requires use_revised_decoder=True.")
        if self.segmentation_mode == "binary" and self.use_revised_decoder:
            raise ValueError("DistillSAM binary training uses the original SAM mask decoder; omit --use-revised-decoder.")

        self.sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
        self.sam.eval()
        if freeze_sam:
            self._freeze_sam()

        self.adapter_layers = adapter_layers or [4, 8, 12]
        self.adapter = self._build_adapter(
            adapter_type=adapter_type,
            pvt_pretrained=pvt_pretrained,
            adapter_layers=self.adapter_layers,
            token_stride=token_stride,
            num_prompt_tokens=num_prompt_tokens,
            debug_shapes=debug_shapes,
        )

        from .sam_image_encoder_intermediate import SamImageEncoderIntermediate

        self.sam_encoder_wrapper = SamImageEncoderIntermediate(
            self.sam.image_encoder,
            adapter_layers=self.adapter_layers,
        )

        self.revised_decoder = None
        if self.segmentation_mode == "semantic":
            self.revised_decoder = RevisedMaskDecoder(
                sam_mask_decoder=self.sam.mask_decoder,
                num_classes=self.num_classes,
                freeze_sam_decoder=True,
                class_head_hidden_dim=class_head_hidden_dim,
                use_default_mask_tokens=use_default_mask_tokens,
                extra_mask_tokens=extra_mask_tokens,
            )
            self.revised_decoder.debug_shapes = self.debug_shapes

    def _freeze_sam(self) -> None:
        for module in (self.sam.image_encoder, self.sam.prompt_encoder, self.sam.mask_decoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def _build_adapter(
        self,
        adapter_type: str,
        pvt_pretrained: str,
        adapter_layers: List[int],
        token_stride: int,
        num_prompt_tokens: int,
        debug_shapes: bool,
    ) -> nn.Module:
        if adapter_type != "pvt_cross":
            raise ValueError(f"DistillSAM final training supports adapter_type='pvt_cross', got {adapter_type!r}.")
        from .trainable_adapter_pvt_cross import TrainableAdapterPVTCross

        return TrainableAdapterPVTCross(
            pvt_pretrained=pvt_pretrained,
            num_prompt_tokens=num_prompt_tokens,
            adapter_layers=adapter_layers,
            token_stride=token_stride,
            debug_shapes=debug_shapes,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam.image_encoder.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.eval()
        if hasattr(self, "sam_encoder_wrapper"):
            self.sam_encoder_wrapper.eval()
        return self

    def get_trainable_parameters(self):
        params = list(self.adapter.parameters())
        if self.segmentation_mode == "semantic":
            params += list(self.revised_decoder.class_head.parameters())
        return params

    def forward(
        self,
        image: torch.Tensor,
        return_kd_features: bool = False,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings = self._encode_prompts(image)
        image_pe = self.sam.prompt_encoder.get_dense_pe().to(image.device)

        if self.segmentation_mode == "binary":
            decoded = self.decode_prompts(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False,
                return_tokens=return_kd_features,
                return_attn=return_attn,
            )
            logits = F.interpolate(
                decoded["mask_logits_low_res"],
                size=(self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.debug_shapes and not self._printed_shapes:
                print(f"[shape] logits: {tuple(logits.shape)}")
                self._printed_shapes = True
            output = {
                "logits": logits,
                "low_res_logits": decoded["mask_logits_low_res"],
                "iou_predictions": decoded["iou_predictions"],
            }
            if return_kd_features:
                output["student_mask_tokens_out"] = decoded["mask_tokens_out"]
                output["student_dense_prompt_embeddings"] = dense_prompt_embeddings
                output["image_embeddings"] = image_embeddings
                output["image_pe"] = image_pe
                if return_attn:
                    output["student_attention_maps"] = decoded["attention_maps"]
            return output

        revised = self.decode_prompts(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=True,
            return_tokens=return_kd_features,
            return_attn=return_attn,
        )
        semantic_logits = F.interpolate(
            revised["semantic_logits_low_res"],
            size=(self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.debug_shapes and not self._printed_shapes:
            print(f"[shape] semantic_logits: {tuple(semantic_logits.shape)}")
            self._printed_shapes = True

        output = {
            "semantic_logits": semantic_logits,
            "semantic_logits_low_res": revised["semantic_logits_low_res"],
            "mask_logits_low_res": revised["mask_logits_low_res"],
            "class_logits": revised["class_logits"],
            "iou_predictions": revised["iou_predictions"],
        }
        if return_kd_features:
            output["student_mask_tokens_out"] = revised["mask_tokens_out"]
            output["student_dense_prompt_embeddings"] = dense_prompt_embeddings
            output["image_embeddings"] = image_embeddings
            output["image_pe"] = image_pe
            if return_attn:
                output["student_attention_maps"] = revised["attention_maps"]
        return output

    def _encode_prompts(self, image: torch.Tensor):
        encoder_output = self.sam_encoder_wrapper(image)
        image_embeddings = encoder_output["image_embeddings"]
        sparse_prompt_embeddings, dense_prompt_embeddings = self.adapter(
            image=image,
            sam_image_embedding=image_embeddings,
            sam_intermediates=encoder_output["intermediates"],
        )
        return image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings

    def decode_prompts(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        return_tokens: bool = False,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.segmentation_mode == "semantic":
            return self.revised_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
                return_tokens=return_tokens,
                return_attn=return_attn,
            )
        return self._decode_original_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            return_tokens=return_tokens,
            return_attn=return_attn,
        )

    def _decode_original_mask_decoder(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
        return_tokens: bool = False,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        mask_logits_list = []
        iou_predictions_list = []
        mask_tokens_out_list = []
        final_attn_list = []
        for index in range(image_embeddings.shape[0]):
            masks, iou_predictions, mask_tokens_out, attention_maps = self._predict_original_one(
                image_embeddings=image_embeddings[index : index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings[index : index + 1],
                dense_prompt_embeddings=dense_prompt_embeddings[index : index + 1],
                return_attn=return_attn,
            )
            mask_slice = slice(1, None) if multimask_output else slice(0, 1)
            mask_logits_list.append(masks[:, mask_slice, :, :])
            iou_predictions_list.append(iou_predictions[:, mask_slice])
            mask_tokens_out_list.append(mask_tokens_out)
            if return_attn:
                final_attn_list.append(attention_maps["final_attn_token_to_image"])

        output = {
            "mask_logits_low_res": torch.cat(mask_logits_list, dim=0),
            "iou_predictions": torch.cat(iou_predictions_list, dim=0),
        }
        if return_tokens:
            output["mask_tokens_out"] = torch.cat(mask_tokens_out_list, dim=0)
        if return_attn:
            output["attention_maps"] = {
                "final_attn_token_to_image": torch.cat(final_attn_list, dim=0),
            }
        return output

    def _predict_original_one(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        return_attn: bool = False,
    ):
        decoder = self.sam.mask_decoder
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
        return masks, iou_predictions, mask_tokens_out, attention_maps

    def _transformer_forward_with_final_attn(
        self,
        transformer,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ):
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
        attn_out, final_attn = RevisedMaskDecoder._attention_forward_with_probs(
            transformer.final_attn_token_to_image,
            q=q,
            k=k,
            v=keys,
        )
        queries = queries + attn_out
        queries = transformer.norm_final_attn(queries)
        return queries, keys, final_attn
