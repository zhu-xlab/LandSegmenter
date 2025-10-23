export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=2

subset_rate=1
max_dataset_size=$(awk "BEGIN {print 15000 * $subset_rate}")

python -u train_scripts/train_baseline.py \
    --dataset_name dynamicearthnet openearthmap sbtn worldcover iran ghsl nlcd usfs \
    --subset_proportion $subset_rate \
    --max_dataset_size $max_dataset_size \
    --save_dir Results \
    --batch_size 12 \
    --test_batch_size 10 \
    --num_workers 8 \
    --use-auxiliary-decoder \
    --dofa-type vit_base_patch16 \
    --dofa-ckpt-path checkpoints/DOFA_ViT_base_e100.pth \
    --dofa-unfix-weights \
    --dofa-lr-scale 0.1 \
    --ckpt-path-clip checkpoints/RS5M_ViT-B-32.pt \
    --accelerator gpu \
    --slurm \
    --epochs 50 \
    --optimizer adamw \
    --learning_rate 1e-4 \
    --load-decoder-weights \
    --focal_loss_weight 0 \
    --cross_entropy_loss_weight 1 \
    --dice_loss_weight 1 \
    --use_multiclass_loss \
    --scheduler_type cos \
    --scheduler_values 0 50 1e-5 \
    --project_name landsegmenter \
    --entity_name username \
    --wandb_mode offline \
    --display_interval 5 \
    --save_interval 10 \
    --seed 42 \
    --use_binary_class_scale \
    --mute-exist-loss