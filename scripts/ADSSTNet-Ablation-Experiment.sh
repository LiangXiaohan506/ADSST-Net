##===================================================== Model Structure Ablation =====================================================##
########################################## EYE OPEN BEFOR no bimamba NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noBi \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --no-bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noBi \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --no-bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no space NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noSp \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noSp \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no sample_adapt NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noSamAda \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --no-sample_adapt_A0 --no-sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noSamAda \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --no-sample_adapt_A0 --no-sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no sample_adapt A0 NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noSamAdaA0 \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --no-sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noSamAdaA0 \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --no-sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no sample_adapt Fre NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noSamAdaFre \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --no-sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noSamAdaFre \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --no-sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no differential mechanism NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noDifMech \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noDifMech \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no TSBiMambaBlock NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noTSBMB \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noTSBMB \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no GraphLayerPearson NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noGLP \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noGLP \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no AdaptiveSpectralBlock NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noASB \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noASB \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no Downsample NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noDownSam \
 --e_layers 6 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noDownSam \
 --e_layers 6 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################### EYE OPEN BEFOR no Layer Connection NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noLayCon \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noLayCon \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no Comparative Loss NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noComLoss \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noComLoss \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR MAMBAminusGCN Loss NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-MAMBAminusGCN \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-MAMBAminusGCN \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR no topk_A0 Loss NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-noTopkA0 \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 20 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-noTopkA0 \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 20 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR A0 random Loss NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-randomA0 \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-randomA0 \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR supConNoUni subInvertsamSub Loss NCvsMCIvsAD NCvsMCI ##########################################
## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-supConSamSub \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-supConSamSub \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --no-space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR NCvsMCIvsAD Augmentation one Frequency Band Mixing delta/alpha/gamma no Random Crop PLI ##########################################
# ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-oneFreBanDAGnoRanCrop-PLI \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

## ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --use_augmentation --augmentations FreBandMix \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-oneFreBanDAGnoRanCrop-PLI \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental


########################################## EYE OPEN BEFOR noAugmentation NCvsMCIvsAD NCvsMCI ##########################################
# ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1,2 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCIvsAD-noAugmentation \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

# ADSSTNet
python \
 -u run.py \
 --seed 38 \
 --task_name supervised \
 --is_training 0 \
 --model ADSSTNet \
 --data CAEEG --root_path ./dataset/CAEEG3_1.0s_0.0_EO_Aug/ \
 --selected_classes 0,1 \
 --project_name CAEEG3_1s_00_EO_Aug-ADSSTNet-NCvsMCI-noAugmentation \
 --downsample_min_len 4 \
 --d_model 20 \
 --expand 2 --d_state 16 --dt_rank 2 --d_conv_in 4 --d_conv 4 --d_conv_out 4 --bimamba --space \
 --dim_UVA 4 --topk_A0 4 \
 --embedding_dim 32 \
 --sample_adapt_A0 --sample_adapt_Fre \
 --projection_dim 128 \
 --group_size 16 --batch_size 256 \
 --itr 5 \
 --learning_rate 0.001 \
 --weight_decay 0.01 --label_smoothing 0.05 \
 --dropout_head 0.7 \
 --lambda_instance 0.5 --lambda_subject 0.5 --supcon_alpha 0.2 --layer_exp_base 0.8 \
 --lambda_regularization 1.0 \
 --train_epochs 200 --patience 15 \
 --warmup_epochs 5 \
 --subject_trim_ratio 0.2 \
 --optimizer AdamW \
 --scheduler CosineAnnealingLR \
 --swa \
 --visualization_training Experimental

