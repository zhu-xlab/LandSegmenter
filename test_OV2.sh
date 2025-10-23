export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=2

test_weight_path=checkpoints/LandSegmenter-aux-epoch=49.ckpt

# python -u ProxyCLIP/test_from_lmdb_pc.py \
#     --test_dataset_names loveda \
#     --test_batch_sizes 20 \
#     --num_workers 8 \
#     --use-auxiliary-decoder \
#     --dofa-type vit_base_patch16 \
#     --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
#     --accelerator gpu \
#     --test_only \
#     --test_weight_path $test_weight_path \
#     --test_use_multiple_prompts \
#     --vfm_model weak \
#     --weak_feature_type encoder_embed \
#     --clip_slide_stride 112 \
#     --clip_slide_crop 336 


python -u ProxyCLIP/test_from_lmdb_fusion.py \
    --test_dataset_names loveda \
    --test_batch_sizes 20 \
    --num_workers 8 \
    --use-auxiliary-decoder \
    --dofa-type vit_base_patch16 \
    --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
    --accelerator gpu \
    --test_only \
    --test_weight_path $test_weight_path \
    --test_use_multiple_prompts \
    --vfm_model weak \
    --weak_feature_type encoder_embed \
    --clip_slide_stride 112 \
    --clip_slide_crop 336 

# python -u ProxyCLIP/test_from_lmdb_pc.py \
#     --test_dataset_names loveda \
#     --test_batch_sizes 20 \
#     --num_workers 8 \
#     --use-auxiliary-decoder \
#     --dofa-type vit_base_patch16 \
#     --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
#     --accelerator gpu \
#     --test_only \
#     --test_weight_path $test_weight_path \
#     --test_use_multiple_prompts \
#     --vfm_model sam2 \
#     --weak_feature_type encoder_embed \
#     --clip_slide_stride 112 \
#     --clip_slide_crop 336 

# python -u ProxyCLIP/test_from_lmdb_pc.py \
#     --test_dataset_names potsdam \
#     --test_batch_sizes 10 \
#     --num_workers 8 \
#     --use-auxiliary-decoder \
#     --dofa-type vit_base_patch16 \
#     --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
#     --accelerator gpu \
#     --seed 42 \
#     --test_only \
#     --test_weight_path $test_weight_path \
#     --test_use_multiple_prompts \
#     --vfm_model sam2 \
#     --weak_feature_type encoder_embed \
#     --clip_slide_stride 112 \
#     --clip_slide_crop 336 