import argparse
import os
import torch
from exp.exp_supervised import Exp_supervised
import random
import numpy as np
from argparse import Namespace
from utils.tools import compute_avg_std



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ADDistinguishNet")

    # basic config
    parser.add_argument("--seed",                   type=int, default=44,                     help="random seed")
    parser.add_argument('--task_name',              type=str, default='pretraining',           help='task name, options:[supervised]')
    parser.add_argument("--is_training",            type=int, default=1,                      help="status")
    parser.add_argument('--model',                  type=str, default='ADSSTNet',              help='model name, options: [ADSSTNet, EEGNet, Conformer, ADformer]')
    parser.add_argument("--project_name",           type=str, default="ADFTD3_2s_05-ADSSTNet_Aug", help="project name")
    parser.add_argument('--checkpoints',            type=str, default='./checkpoints/',       help='location of model checkpoints')
    parser.add_argument('--visualization_training', type=str, default='Dubeg',                help='options: [Dubeg, Control, Parameter_Adjustment, Experimental]')


    # data config
    parser.add_argument("--data",               type=str,  default="ADFTD", help="dataset type")
    parser.add_argument("--root_path",          type=str,  default="./dataset/ADFTD3_2.0s_0.0_Aug/", help="root path of the data file")
    parser.add_argument("--selected_classes",   type=str,  default=None,
                        help="comma-separated class indices to use, e.g. '0,1' or '0,2'. None means all classes (CAEEG)")
    parser.add_argument("--group_size",         type=int,  default=8,       help="How many samples are in a mini-batch for each subject? [>2, 4, 8, 16, 32]")
    parser.add_argument("--num_class",          type=int,  default=3,       help="number of classes")
    parser.add_argument("--enc_in",             type=int,  default=7,       help="number of channels, embedding output channel dimension")
    parser.add_argument("--seq_len",            type=int,  default=128,     help="input sequence length")
    parser.add_argument("--train_mean_values",  type=int,  default=None,    help="mean of training set samples")
    parser.add_argument("--train_std_values",   type=int,  default=None,    help="std of training set samples")
    parser.add_argument("--sampling_rate",      type=int,  default=128,     help="frequency sampling rate")
    parser.add_argument("--use_augmentation",   action=argparse.BooleanOptionalAction, default=False,     help="whether to use augmentation")
    
    # ADSSTNet params
    parser.add_argument("--d_model",      type=int,     default=128,    help="number of channels for ADSSTNet, embedding output channel dimension")
    parser.add_argument('--expand',       type=int,     default=2,      help='expansion factor for Mamba')
    parser.add_argument("--e_layers",     type=int,     default=0,      help="num of encoder layers")
    parser.add_argument("--d_state",      type=int,     default=16,     help="dimension of state")
    parser.add_argument("--d_conv_in",    type=int,     default=4,      help="dimension of input conv")
    parser.add_argument("--d_conv",       type=int,     default=4,      help="dimension of conv")
    parser.add_argument("--d_conv_out",   type=int,     default=4,      help="dimension of output conv")
    parser.add_argument("--dt_rank",      type=int,     default=1,      help="dimension of dt rank")
    parser.add_argument("--bimamba",      action=argparse.BooleanOptionalAction,      default=True,  help="whether to use bimamba")
    parser.add_argument("--space",        action=argparse.BooleanOptionalAction,      default=True,  help="whether to use space")
    parser.add_argument("--se_reduction", type=int,     default=16,     help="reduction factor for SE")
    parser.add_argument("--dropout_head", type=float,   default=0.3,    help="dropout classification head")
    parser.add_argument("--dim_UVA",      type=int,     default=16,     help="dimension of low-rank factors")
    parser.add_argument("--A0",           type=str,     default=None,   help="A0")
    parser.add_argument("--topk_A0",      type=int,     default=6,      help="topk")
    parser.add_argument("--sample_adapt_A0",            action=argparse.BooleanOptionalAction, default=False, help="whether to enable sample correlation plot fine-tuning")
    parser.add_argument("--embedding_dim",type=int,     default=0,      help="gcn embedding_dim")
    parser.add_argument("--projection_dim",type=int,    default=128,    help="contrastive learning projection dimension")
    parser.add_argument("--sample_adapt_Fre",           action=argparse.BooleanOptionalAction, default=False, help="whether to enable sample correlation plot fine-tuning")
    parser.add_argument("--num_heads",    type=int,     default=4,      help="number of attention heads")

    # ADformer params
    parser.add_argument("--up_dim_list",        type=str,   default="38,76", help="a list of up dimension factor used in ADformer")
    parser.add_argument("--no_temporal_block",  action="store_true",         help="whether to use temporal block in encoder", default=False)
    parser.add_argument("--no_spatial_block",   action="store_true",         help="whether to use channel block in encoder", default=False)
    parser.add_argument("--patch_len_list",     type=str,   default="2,2,2,4,4,4,16,16,16,16,32,32,32,32,32", help="a list of patch len used in Medformer")
    parser.add_argument("--n_heads",            type=int,   default=8,       help="num of heads")
    parser.add_argument("--output_attention",   action="store_true",         help="whether to output attention in encoder")
    parser.add_argument("--no_inter_attn",      action="store_true",         default=False,    help="whether to use inter-attention in encoder, using this argument means not using inter-attention")
    parser.add_argument('--activation',         type=str,   default='gelu',  help='activation')

    # Medformer params
    parser.add_argument("--single_channel",     action="store_true",         default=False, help="whether to use single channel patching for Medformer")

    # TSLANet params
    parser.add_argument("--patch_size",      type=int,  default=8,    help="patch size")
    parser.add_argument("--use_icb",         type=bool, default=True, help="whether to use ICB")
    parser.add_argument("--use_asb",         type=bool, default=True, help="whether to use ASB")
    parser.add_argument("--adaptive_filter", type=bool, default=True, help="whether to use adaptive filter")

    # MedGNN params
    parser.add_argument('--resolution_list', type=str,   default="2,4,6,8")
    parser.add_argument('--nodedim',         type=int,   default=10)

    # EEGNet params
    parser.add_argument('--dropout', type=float, default=0.1, help='EEGNet TemporalSpatialConv dropout')

    # Conformer params
    parser.add_argument("--d_ff",    type=int,   default=256, help="dimension of fcn, encoderLayer channel expansion")
    parser.add_argument("--factor",  type=int,   default=1,   help="attn factor")

    # optimization
    parser.add_argument("--cross_val",                 type=str,      default="mccv",   help="options: [mccv]")
    parser.add_argument("--itrs",                      type=int,      default=1,        help="experiments times")
    parser.add_argument("--batch_size",                type=int,      default=32,       help="batch size of train input data.")
    parser.add_argument("--num_workers",               type=int,      default=0,        help="data loader num workers")
    parser.add_argument("--train_epochs",              type=int,      default=1,        help="train epochs")
    parser.add_argument("--patience",                  type=int,      default=10,       help="early stopping patience")
    parser.add_argument("--optimizer",                 type=str,      default='Adam',   help="optimizer function")
    parser.add_argument("--learning_rate",             type=float,    default=0.0001,   help="optimizer learning rate")
    parser.add_argument("--weight_decay",              type=float,    default=0,        help="weight decay (L2 penalty)")
    parser.add_argument("--label_smoothing",           type=float,    default=0,        help="label smoothing")
    parser.add_argument("--scheduler",                 type=str,      default=None,     help="scheduler function, [CosineAnnealingLR, ConstantLR]")
    parser.add_argument("--swa",                       action=argparse.BooleanOptionalAction, default=True,     help="use stochastic weight averaging")
    parser.add_argument("--secondStep",                action=argparse.BooleanOptionalAction, default=False,    help="whether to use second step for model training")
    parser.add_argument("--second_epochs",             type=int,      default=100,      help="second step train epochs")
    parser.add_argument("--warmup_epochs",             type=int,      default=0,        help="epochs with only CE loss (no consistency)")
    parser.add_argument("--subject_trim_ratio",        type=float,    default=0.2,      help="subject-level: drop lowest-confidence fraction of windows before majority vote (0=no trim)")
    parser.add_argument("--supcon_temp",               type=float,    default=1.0,      help="temperature for contrastive loss")
    parser.add_argument("--supcon_alpha",              type=float,    default=0.8,      help="weight for instance contrastive loss")
    parser.add_argument("--lambda_instance",           type=float,    default=0.0,      help="weight for instance contrastive loss")
    parser.add_argument("--lambda_subject",            type=float,    default=0.0,      help="weight for subject contrastive loss")
    parser.add_argument("--layer_exp_base",            type=float,    default=0.0,      help="multi-level weight decay rate")
    parser.add_argument("--downsample_min_len",        type=int,      default=None,     help="minimum length to downsample")
    parser.add_argument("--lambda_regularization",     type=float,    default=None,     help="weight for regularization loss")

    parser.add_argument("--augmentations",             type=str,      default="none",
                        help="a comma-seperated list of augmentation types (none, jitter or scale). "
                             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",)

    # GPU
    parser.add_argument("--use_gpu",        action="store_true",        default=True,   help="use gpu")
    parser.add_argument("--gpu",            type=int,   default=0,      help="gpu")
    parser.add_argument("--use_multi_gpu",  action="store_true",        default=False,  help="use multiple gpus")
    parser.add_argument("--devices",        type=str,   default="0,1",  help="device ids of multiple gpus")



    parser.add_argument("--Overlap_Ratio_Analysis", type=bool, default=False, help="whether to use overlap-ratio analysis")

    args = parser.parse_args()
    # 解析 selected_classes：如 "0,1" -> [0, 1]，None 或空字符串表示使用全部类别
    if getattr(args, 'selected_classes', None) and args.selected_classes.strip():
        args.selected_classes = [int(x.strip()) for x in args.selected_classes.split(',')]
    else:
        args.selected_classes = None
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print("{}".format(args))

    if args.task_name == "supervised":
        Exp = Exp_supervised
    elif args.task_name == "pretraining":
        Exp = Exp_contrastive
    elif args.task_name == "fine-tuning":
        Exp = Exp_finetuning
    else:
        raise ValueError(f"Invalid task name: {args.task_name}")

    sample_test_metrics_dict_list = []
    subject_test_metrics_dict_list = []

    for itr in range(args.itrs):
        args.seed += 1
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        config = Namespace(
            project_name=args.project_name,
            itr='itr_'+f"{itr:03d}",
            train_epochs=args.train_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            ckpt_path='checkpoint.pt',
            visualization_training=args.visualization_training
        )

        exp = Exp(args, config)  # set experiments

        if args.is_training:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(itr+1))
            exp.train(config)

        if args.task_name == "supervised" or args.task_name == "fine-tuning":
            print(">>>>>>>>>>>>>>>>>>>>>>>>>> start testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(itr+1))
            sample_test_metrics_dict, subject_test_metrics_dict = exp.test(config=config)
            sample_test_metrics_dict_list.append(sample_test_metrics_dict)
            subject_test_metrics_dict_list.append(subject_test_metrics_dict)
        torch.cuda.empty_cache()
    compute_avg_std(args, sample_test_metrics_dict_list, subject_test_metrics_dict_list)
