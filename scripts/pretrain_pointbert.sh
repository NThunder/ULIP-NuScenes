CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --model ULIP_PointBERT --npoints 512 --lr 3e-3 --output-dir ./outputs/reproduce_pointbert_8kpts_version_dataset
