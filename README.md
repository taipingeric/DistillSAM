## Distillation-SAM: Knowledge Distillation-Based Auto-Prompt Embedding Learning for Surgical Image Segmentation

### Model Download

Place the pretrained [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and [PVT-tiny checkpoint](https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth) at:

```text
checkpoints/sam_vit_b_01ec64.pth
checkpoints/pvt/pvt_tiny.pth
```
### Training
Then start multi-GPU semantic DistillSAM training with DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --use_env \
  distsam/tools/train_distillsam.py \
  --config distsam/configs/EndoVisSub2017.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --pvt-pretrained checkpoints/pvt/pvt_tiny.pth \
  --adapter-layers 4,8,12 \
  --token-stride 1 \
  --segmentation-mode semantic \
  --use-revised-decoder \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --num-workers 4 \
  --device cuda \
  --save-dir outputs/EndoVisSub2017
```
##### Contact Us
```text
BibTeX
@ARTICLE{TMI26,
  author={Tang, Jiyang and Han, Hu and Shan, Shiguang and Chen, Xilin},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Distillation-SAM: Knowledge Distillation Based Auto-prompt Embedding Learning for Surgical Image Segmentation}, 
  year={2026}
}
```
