#!/bin/bash


python ./run_hybrid_clip.py \
    --output_dir "./Arabic-clip-vit-large-patch14" \
    --overwrite_output_dir \
    --tokenizer_name="asafaya/bert-large-arabic" \
    --train_file="coco_dataset/train_dataset.json" \
    --validation_file="coco_dataset/valid_dataset.json" \
    --do_train --do_eval \
    --num_train_epochs="20" --max_seq_length 90 \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="0.00001" --warmup_ratio 0.1 --weight_decay 0.0 \
    --preprocessing_num_workers 32 \
    --log_comet \
    --exp_name large_clip_patch16_v1 \
    --eval_when 1 \
    --text_model_name_or_path="asafaya/bert-large-arabic" \
    --vision_model_name_or_path="google/vit-large-patch32-384" \
    
    
    --push_to_hub


    #--run_from_checkpoint /home/giuseppe/models/pretraining_on_v4/26/

#    --freeze_backbones    # freezes both models except from the reprojections layers
#    --run_from_checkpoint /home/giuseppe/models/pretraining_on_v4/26/ \
#    --push_to_hub


# python run_clip_pytorch.py \
#     --output_dir ./clip-bert-finetuned \
#     --model_name_or_path ./clip-roberta \
#     --freeze_vision_model=True\
#     --preprocessing_num_workers 32 \
#     --remove_unused_columns=False \
#     --train_file="coco_dataset/train_dataset.json" \
#     --validation_file="coco_dataset/valid_dataset.json" \
#     --do_train  --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --caption_column captions \
#     --image_column image_path \


