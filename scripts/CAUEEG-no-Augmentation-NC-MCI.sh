########################################## EYE OPEN ##########################################
# EEGNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCI-noAugmentation \
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

# Conformer
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCI-noAugmentation \
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

# ADformer
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model ADformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --augmentations flip,frequency,jitter,mask,channel,drop \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-ADformer-NCvsMCI-noAugmentation \
 --group_size 16 \
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

# Medformer
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model Medformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --augmentations flip,frequency,jitter,mask,channel,drop \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Medformer-NCvsMCI-noAugmentation \
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

# TSLANet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCI-noAugmentation \
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

# MedGNN
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model MedGNN \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --augmentations flip,frequency,jitter,mask,channel,drop \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-MedGNN-NCvsMCI-noAugmentation \
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

# STCGRU
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCI-noAugmentation \
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

# CodimNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCI-noAugmentation \
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

