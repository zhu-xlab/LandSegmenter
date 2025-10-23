export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=$1

dataset_proportion=0.1
test_weight_path=checkpoints/LandSegmenter-aux-epoch=49.ckpt

python -u train_scripts/train_fine_tune.py \
    --dataset_name $2 \
    --dataset_proportion $dataset_proportion \
    --save_dir Results \
    --batch_size 10 \
    --test_batch_size 10 \
    --num_workers 8 \
    --dofa-type vit_base_patch16 \
    --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
    --accelerator gpu \
    --epochs 30 \
    --optimizer adamw \
    --learning_rate 5e-4 \
    --focal_loss_weight 0 \
    --cross_entropy_loss_weight 1 \
    --dice_loss_weight 1 \
    --scheduler_type cos \
    --scheduler_values 0 30 \
    --project_name landsegmenter \
    --entity_name username \
    --wandb_mode offline \
    --display_interval 5 \
    --save_interval 30 \
    --seed 1 \
    --use_binary_loss \
    --fine_tune_checkpoint_path $test_weight_path \
    --freeze_backbones_only