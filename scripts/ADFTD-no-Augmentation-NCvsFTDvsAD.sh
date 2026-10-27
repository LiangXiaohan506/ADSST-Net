## EEGNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-EEGNet-NCvsFTDvsAD-noAugmentation \
 --dropout 0.1 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.0001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## Conformer
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-Conformer-NCvsFTDvsAD-noAugmentation \
 --d_model 128 --d_ff 256 --e_layers 12 --n_heads 8 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.0001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADformer
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model ADformer \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --augmentations flip,frequency,jitter,mask,channel,drop \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-ADformer-NCvsFTDvsAD-noAugmentation \
 --patch_len_list 8,8,8,16,16,16 \
 --up_dim_list 76,76,76 \
 --e_layers 6 --n_heads 8 --d_model 128 --d_ff 256 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.0001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## Medformer
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model Medformer \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --augmentations flip,frequency,jitter,mask,channel,drop \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-Medformer-NCvsFTDvsAD-noAugmentation \
 --e_layers 6 --d_model 128 --d_ff 256 \
 --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.0001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## TSLANet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-TSLANet-NCvsFTDvsAD-noAugmentation \
 --embedding_dim 128 --e_layers 2 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## MedGNN
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model MedGNN \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --augmentations flip,frequency,jitter,mask,channel,drop \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-MedGNN-NCvsFTDvsAD-noAugmentation \
 --e_layers 12 --n_heads 8 --d_model 128 --d_ff 256 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.0001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## STCGRU
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-STCGRU-NCvsFTDvsAD-noAugmentation \
 --dropout 0.1 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.003 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## CodimNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data ADFTD --root_path ./dataset/ADFTD3_1.0s_0.0_Aug/ \
 --selected_classes 0,1,2 \
 --project_name ADFTD3_1s_00_Aug-CodimNet-NCvsFTDvsAD-noAugmentation \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --train_epochs 200 --patience 15 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental