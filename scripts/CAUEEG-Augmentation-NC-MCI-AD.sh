########################################## EYE OPEN Augmentation one Frequency Band Mixing delta/alpha/gamma no Random Crop ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAUEEG --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop \
 --downsample_min_len 4 \
 --d_model 19 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 64 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 64 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.5 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


## EEGNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop \
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
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-oneFreBanDAGnoRanCrop \
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

## TSLANet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop \
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

## STCGRU
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-oneFreBanDAGnoRanCrop \
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

## CodimNet
python \
 -u run.py \
 --seed 41 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop \
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


########################################## EYE OPEN BEFOR Augmentation three Frequency Band Mixing delta/alpha/gamma no Random Crop ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-threeFreBanDAGnoRanCrop \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-threeFreBanDAGnoRanCrop \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-threeFreBanDAGnoRanCrop \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-threeFreBanDAGnoRanCrop \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-threeFreBanDAGnoRanCrop \
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


########################################## EYE OPEN BEFOR Augmentation one Frequency Band Mixing delta/theta/alpha/beta/gamma no Random Crop ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-oneFreBanDTABGnoRanCrop \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-oneFreBanDTABGnoRanCrop \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-oneFreBanDTABGnoRanCrop \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-oneFreBanDTABGnoRanCrop \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-oneFreBanDTABGnoRanCrop \
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


########################################## EYE OPEN BEFOR Augmentation one Frequency Band Mixing theta/beta no Random Crop ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-oneFreBanTBnoRanCrop \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-oneFreBanTBnoRanCrop \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-oneFreBanTBnoRanCrop \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-oneFreBanTBnoRanCrop \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-oneFreBanTBnoRanCrop \
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


########################################## EYE OPEN BEFOR Augmentation Jitter ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations jitter \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-Jitter \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations jitter \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-Jitter \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations jitter \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-Jitter \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations jitter \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-Jitter \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations jitter \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-Jitter \
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


########################################## EYE OPEN BEFOR Augmentation Drop ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations drop \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-Drop \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations drop \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-Drop \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations drop \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-Drop \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations drop \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-Drop \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations drop \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-Drop \
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


######################################### EYE OPEN BEFOR Augmentation Flip ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations flip \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-Flip \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations flip \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-Flip \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations flip \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-Flip \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations flip \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-Flip \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations flip \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-Flip \
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


######################################### EYE OPEN BEFOR Augmentation Frequency ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations frequency \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-Frequency \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations frequency \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-Frequency \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations frequency \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-Frequency \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations frequency \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-Frequency \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations frequency \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-Frequency \
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


######################################### EYE OPEN BEFOR Augmentation Mask ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations mask \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-Mask \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations mask \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-Mask \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations mask \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-Mask \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations mask \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-Mask \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations mask \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-Mask \
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


######################################### EYE OPEN BEFOR Augmentation Channel ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations channel \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-Channel \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations channel \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-Channel \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations channel \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-Channel \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations channel \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-Channel \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations channel \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-Channel \
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


########################################## EYE OPEN BEFOR Augmentation Frequency Mixing ##########################################
## EEGNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model EEGNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-EEGNet-NCvsMCIvsAD-FreMixing \
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
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model Conformer \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-Conformer-NCvsMCIvsAD-FreMixing \
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

## TSLANet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model TSLANet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-TSLANet-NCvsMCIvsAD-FreMixing \
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

## STCGRU
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model STCGRU \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-STCGRU-NCvsMCIvsAD-FreMixing \
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

## CodimNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model CodimNet \
 --data CAUEEG \
 --root_path ./dataset/CAUEEG_Dementia_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreMix \
 --selected_classes 0,1,2 \
 --project_name CAUEEG_Dementia_1s_00_EO_Aug-CodimNet-NCvsMCIvsAD-FreMixing \
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