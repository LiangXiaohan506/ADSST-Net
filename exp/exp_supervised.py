import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import random
import datetime
import wandb
import visdom
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from utils.tools import calculate_subject_level_metrics
from utils.tools import compute_pearson_A0_subject_mean, generate_random_A0, compute_pli_A0_subject_mean
from utils.losses import kl_reg_group_mean_multiscale, wasserstein_reg_group_mean_multiscale
from utils.losses import crossentropy_loss_multiscale
from utils.losses import hierarchical_supcon_loss_multiscale_augmentation
from layers.Augmentation import frequency_mixing, frequency_band_mixing
from layers.Augmentation import get_augmentation
warnings.filterwarnings("ignore")


class Exp_supervised(Exp_Basic):
    def __init__(self, args, config=None):
        self.config = config  # 必须在 super().__init__ 之前设置，因为父类 __init__ 会调用 _build_model
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        if self.swa:
            self.swa_model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def _build_model(self):
        # model input depends on data
        train_data, _ = self._get_data(flag='TRAIN')
        self.args.pred_len = 0
        self.args.seq_len = train_data.max_seq_len  # redefine seq_len 数据长度
        self.args.embedding_dim = self.args.seq_len if self.args.embedding_dim == 0 else self.args.embedding_dim
        self.args.enc_in = train_data.X.shape[-1]  # redefine enc_in 通道数
        self.args.num_class = len(np.unique(train_data.y[:, 0]))  # redefine num_class
        self.args.A0 = compute_pearson_A0_subject_mean(
            train_data, enc_in=self.args.enc_in, use_abs=True, topk=self.args.topk_A0, sym_mode="max",
            eps=1e-8, row_normalize=True, clip_negative=True, dataset_name=self.args.data)
        # self.args.A0 = compute_pli_A0_subject_mean(
        #     self.args.root_path, train_data.train_ids, topk=self.args.topk_A0, sym_mode="max",
        #     eps=1e-8, row_normalize=True, clip_negative=True, dataset_name=self.args.data)
        # self.args.A0 = generate_random_A0(self.args.enc_in, self.args.topk_A0, seed=self.args.seed)
        model = self.model_dict[self.args.model].Model(self.args).float()  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 仅在训练时保存模型统计与初始化参数到 init_params.txt
        # if self.args.is_training == 0:
        #     save_dir = os.path.join(self.args.checkpoints, self.args.project_name, self.config.itr)
        #     os.makedirs(save_dir, exist_ok=True)
        #     init_params_path = os.path.join(save_dir, "init_params.txt")
        #     self._save_init_params_txt(model, init_params_path)
        return model

    def _get_data(self, flag, log=None):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag, log)
        return data_set, data_loader

    def _select_optimizer(self, optimType='Adam'):
        if optimType == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            return model_optim
        elif optimType == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            return model_optim
        elif optimType == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            return model_optim
        elif optimType == 'RMSprop':
            model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            return model_optim
        elif optimType == 'Adagrad':
            model_optim = optim.Adagrad(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            return model_optim
        else:
            print(f"Invalid optimizer: {optimType}")
            return None

    def _select_scheduler(self, model_optim, schedulerType=None):
        if schedulerType == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs, eta_min=self.args.learning_rate/10)
            return scheduler
        elif schedulerType == 'ConstantLR':
            scheduler = torch.optim.lr_scheduler.ConstantLR(model_optim)
            return scheduler
        else:
            return None

    # def _select_criterion(self, labelNum=None):
    #     criterion = {'TRAIN': None, 'VAL': None}
    #     if labelNum is not None: # 处理类不平衡的问题
    #         weights_normalized = ((1.0/labelNum.float()) * labelNum.sum().item() / len(labelNum)).to(self.device)
    #         criterion['TRAIN'] = nn.CrossEntropyLoss(weight=weights_normalized, label_smoothing=self.args.label_smoothing)
    #     else:
    #         criterion['TRAIN'] = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
    #     criterion['VAL'] = nn.CrossEntropyLoss(reduction='sum')
    #     return criterion

    def _select_criterion(self, labelNum=None):
        if labelNum is not None: # 处理类不平衡的问题
            weights_normalized = ((1.0/labelNum.float()) * labelNum.sum().item() / len(labelNum)).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_normalized, label_smoothing=self.args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        return criterion

    #%%
    def evaluation(self, evaluation_loader, criterion, eval_model='validation'):
        loss_dict = {'total': [], 'ce': [], 'regularization': [], 'instance': [], 'subject': []}
        preds, trues, ids = [], [], []

        if self.swa: self.swa_model.eval()
        else: self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label_id, _) in enumerate(evaluation_loader):
                label = label_id[:, 0].to(self.device)
                subject_id = label_id[:, 1].to(self.device)
                batch_x_ori = batch_x[:, 0, :, :].float().to(self.device)

                if self.swa:
                    if self.args.model == "ADSSTNet": # , _, _, _, _
                        layer_out, logits, _, _, _, _ = self.swa_model(batch_x_ori, None, None, None, model_state=eval_model)
                    elif self.args.model == "CodimNet":
                        logits, nd_loss = self.model(batch_x_ori, None, None, None)
                    else:
                        logits = self.swa_model(batch_x_ori, None, None, None)
                else:
                    if self.args.model == "ADSSTNet": # , _, _, _, _
                        layer_out, logits, _, _, _, _ = self.model(batch_x_ori, None, None, None, model_state=eval_model)
                    elif self.args.model == "CodimNet":
                        logits, nd_loss = self.model(batch_x_ori, None, None, None)
                    else:
                        logits = self.model(batch_x_ori, None, None, None)

                if eval_model == 'validation':
                    loss_ce = criterion(logits, label.long())
                    # loss_ce = crossentropy_loss_multiscale(layer_out, logits, label, criterion, layer_exp_base=0.5)
                    loss_dict['ce'].append(loss_ce.item())
                    if self.args.lambda_regularization is not None and self.add_regularization:
                        # loss_reg = kl_reg_group_mean_multiscale(layer_out, None, subject_id.long(), label.long(), self.args.lambda_ss, layer_exp_base=0.5)
                        # loss_wasserstein_reg = wasserstein_reg_group_mean_multiscale(layer_out, None, subject_id.long(), None, self.args.lambda_ss)
                        loss_instance, loss_subject, loss_reg = hierarchical_supcon_loss_multiscale_augmentation(
                            layer_out, label, subject_id, self.args.lambda_instance, self.args.lambda_subject,
                            supcon_temp=self.args.supcon_temp, supcon_alpha=self.args.supcon_alpha, layer_exp_base=self.args.layer_exp_base)
                        loss_dict['instance'].append(loss_instance.item())
                        loss_dict['subject'].append(loss_subject.item())
                        loss_dict['regularization'].append(loss_reg.item())
                        loss_total = loss_ce + self.args.lambda_regularization * loss_reg
                    else:
                        loss_instance = torch.tensor([0.0]).to(self.device)
                        loss_subject = torch.tensor([0.0]).to(self.device)
                        loss_reg = torch.tensor([0.0]).to(self.device)
                        loss_dict['instance'].append(loss_instance.item())
                        loss_dict['subject'].append(loss_subject.item())
                        loss_dict['regularization'].append(loss_reg.item())
                        if self.args.model == "CodimNet":
                            loss_total = loss_ce + nd_loss
                        else:
                            loss_total = loss_ce
                    loss_dict['total'].append(loss_total.item())

                pred_logits = logits[-1] if isinstance(logits, (tuple, list)) else logits
                preds.append(pred_logits.detach())
                trues.append(label.detach())
                ids.append(subject_id.detach())

        if eval_model == 'validation':
            loss_dict['ce'] = np.average(loss_dict['ce'])
            loss_dict['instance'] = np.average(loss_dict['instance'])
            loss_dict['subject'] = np.average(loss_dict['subject'])
            loss_dict['regularization'] = np.average(loss_dict['regularization'])
            loss_dict['total'] = np.average(loss_dict['total'])

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        ids = torch.cat(ids, 0)
        probs = torch.nn.functional.softmax(preds, dim=-1)  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            ).float().cpu().numpy()
        )
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        ids = ids.flatten().cpu().numpy()

        sample_metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "k-score": cohen_kappa_score(trues, predictions),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
            "Confusion Matrix": confusion_matrix(trues, predictions)
        }  # sample-level performance metrics

        subject_metrics_dict = calculate_subject_level_metrics(
            predictions, trues, ids, self.args.num_class,
            pred_probs=probs,
            trim_ratio=getattr(self.args, "subject_trim_ratio", 0.2),
        )  # subject-level: trimmed mean 后多数投票

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return loss_dict, sample_metrics_dict, subject_metrics_dict


    #%%
    def train(self, config):
        # 初始化训练可视化工具
        # if config.visualization_training == 'Control' or config.visualization_training == 'Experimental':
        #     nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)
        # elif config.visualization_training == 'Parameter_Adjustment':
        #     vis = visdom.Visdom(env=self.args.project_name)  # 设置环境窗口的名称,如果不设置名称就默认为main
        #     opt_train_loss_total  = {'xlabel': 'epochs', 'ylabel': 'loss',     'title': 'train_loss_total'}
        #     opt_train_loss_ce     = {'xlabel': 'epochs', 'ylabel': 'loss',     'title': 'train_loss_ce'}
        #     opt_train_loss_regularization =   {'xlabel': 'epochs', 'ylabel': 'loss',    'title': 'train_loss_regularization'}
        #     opt_train_acc         = {'xlabel': 'epochs', 'ylabel': 'accuracy', 'title': 'train_acc'}
        #     opt_train_F1          = {'xlabel': 'epochs', 'ylabel': 'F1 score', 'title': 'train_F1'}
        #     opt_eval_loss_total   = {'xlabel': 'epochs', 'ylabel': 'loss',     'title': 'eval_loss_total'}
        #     opt_eval_loss_ce      = {'xlabel': 'epochs', 'ylabel': 'loss',     'title': 'eval_loss_ce'}
        #     opt_eval_loss_regularization =    {'xlabel': 'epochs', 'ylabel': 'loss',    'title': 'eval_loss_regularization'}
        #     opt_eval_acc          = {'xlabel': 'epochs', 'ylabel': 'accuracy', 'title': 'eval_acc'}
        #     opt_eval_F1           = {'xlabel': 'epochs', 'ylabel': 'F1 score', 'title': 'eval_F1'}
        #     # opt_lr              = {'xlabel':'epochs', 'ylabel':'lr',       'title':'learn rate'}
        #     train_acc_window         = vis.line(X=[0], Y=[0.0], opts=opt_train_acc)
        #     train_F1_window          = vis.line(X=[0], Y=[0.0], opts=opt_train_F1)
        #     train_loss_total_window  = vis.line(X=[0], Y=[0.0], opts=opt_train_loss_total)
        #     train_loss_ce_window     = vis.line(X=[0], Y=[0.0], opts=opt_train_loss_ce)
        #     train_loss_regularization_window = vis.line(X=[0], Y=[0.0], opts=opt_train_loss_regularization)
        #     eval_acc_window          = vis.line(X=[0], Y=[0.0], opts=opt_eval_acc)
        #     eval_F1_window           = vis.line(X=[0], Y=[0.0], opts=opt_eval_F1)
        #     eval_loss_total_window   = vis.line(X=[0], Y=[0.0], opts=opt_eval_loss_total)
        #     eval_loss_ce_window      = vis.line(X=[0], Y=[0.0], opts=opt_eval_loss_ce)
        #     eval_loss_regularization_window = vis.line(X=[0], Y=[0.0], opts=opt_eval_loss_regularization)
        #     # lr_window               = vis.line(X=[0], Y=[0.0], opts=opt_lr)

        # 检查并创建 checkpoints 路径
        checkpointsPath = (self.args.checkpoints + self.args.project_name + "/" + config.itr + "/")
        if not os.path.exists(checkpointsPath):
            os.makedirs(checkpointsPath)
        log_write = open(checkpointsPath + "/log.txt", "w", encoding='utf-8')
        info = ">>>>>>>>>>>>>>>>>>>>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(config.itr)
        log_write.write(info + "\n")

        # 加载数据
        train_data, train_loader = self._get_data(flag="TRAIN", log=log_write)
        vali_data, vali_loader   = self._get_data(flag="VAL",   log=log_write)
        test_data, test_loader   = self._get_data(flag="TEST",  log=log_write)
        info = "train dataset: {}\t classes ratio: {}\t vali dataset: {}\t classes ratio: {}\n".format(
            len(train_data), 
            torch.bincount(torch.from_numpy(train_data.y[:, 0]).long()), 
            len(vali_data), 
            torch.bincount(torch.from_numpy(vali_data.y[:, 0]).long())
        )
        print(info)
        log_write.write(info + "\n")

        # 设置早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-3)

        # 选择优化器和调度器
        model_optim = self._select_optimizer(optimType=self.args.optimizer)
        scheduler = self._select_scheduler(model_optim, schedulerType=self.args.scheduler)
        criterion = self._select_criterion(labelNum=torch.bincount(torch.from_numpy(train_data.y[:, 0]).long()))

        # 记录验证集最优值
        best_val = {'f1': -1.0, 'acc': -1.0, 'to_test_f1': -1.0, 'to_test_acc': -1.0, 'to_test_acc_sub': -1.0, 'test_f1': -1.0, 'test_acc': -1.0, 'test_acc_sub': -1.0}
        self.add_regularization = False if self.args.warmup_epochs > 0 else True

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            preds, trues = [], []
            train_loss_total, train_loss_ce, train_loss_regularization = [], [], []
            train_loss_instance, train_loss_subject = [], []

            self.model.train()
            epoch_time = time.time()

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                dynamic_ncols=True,
                leave=True,
            )
            for i, (batch_x, label_id, _) in pbar:
                iter_count += 1
                model_optim.zero_grad()
                label_y = label_id[:, 0].to(self.device)
                subject_id = label_id[:, 1].to(self.device)
                if self.args.use_augmentation:
                    batch_x_ori = batch_x[:, 0, :, :].float().to(self.device)
                    if self.args.augmentations == "RandomCrop":
                        batch_x_aug = batch_x[:, 1, :, :].float().to(self.device)
                    elif self.args.augmentations == "FreBandMix":
                        batch_x_aug = frequency_band_mixing(
                            batch_x[:, 0, :, :].float().to(self.device),
                            sampling_rate=self.args.sampling_rate,
                            label=label_y,
                            subject_id=subject_id,
                            seed=self.args.seed,
                            train_pairing="same_class"
                        ).to(self.device)
                    elif self.args.augmentations == "FreMix":
                        batch_x_aug = frequency_mixing(
                            batch_x[:, 0, :, :].float().to(self.device),
                            label=label_y,
                            subject_id=subject_id,
                            seed=self.args.seed,
                            train_pairing="same_class"
                        ).to(self.device)
                    else:
                        self.augmentation = get_augmentation(self.args.augmentations)
                        batch_x_aug = self.augmentation(batch_x[:, 0, :, :].float().to(self.device))
                    batch_x = torch.cat((batch_x_ori, batch_x_aug), dim=0)
                    label_y = torch.cat((label_y, label_y), dim=0)
                    subject_id = torch.cat((subject_id, subject_id), dim=0)
                else:
                    batch_x = batch_x[:, 0, :, :].float().to(self.device)


                if self.args.model == "ADSSTNet": # , _, _, _, _
                    layer_out, logits, _, _, _, _ = self.model(batch_x, None, None, None)
                elif self.args.model == "CodimNet":
                    logits, nd_loss = self.model(batch_x, None, None, None)
                else:
                    logits = self.model(batch_x, None, None, None)

                loss_ce = criterion(logits, label_y.long())
                # loss_ce = crossentropy_loss_multiscale(layer_out, logits, label_y, criterion, layer_exp_base=0.5)
                train_loss_ce.append(loss_ce.item())
                if self.args.lambda_regularization is not None and self.add_regularization:
                    # loss_reg = kl_reg_group_mean_multiscale(layer_out, None, subject_id.long(), label_y.long(), self.args.lambda_ss, layer_exp_base=0.5)
                    # loss_wasserstein_reg = wasserstein_reg_group_mean_multiscale(layer_out, None, subject_id.long(), None, self.args.lambda_ss)
                    loss_instance, loss_subject, loss_reg = hierarchical_supcon_loss_multiscale_augmentation(
                        layer_out, label_y, subject_id, self.args.lambda_instance, self.args.lambda_subject,
                        supcon_temp=self.args.supcon_temp, supcon_alpha=self.args.supcon_alpha, layer_exp_base=self.args.layer_exp_base)
                    train_loss_instance.append(loss_instance.item())
                    train_loss_subject.append(loss_subject.item())
                    train_loss_regularization.append(loss_reg.item())
                    loss_total = loss_ce + self.args.lambda_regularization * loss_reg
                else:
                    loss_reg = torch.tensor([0.0]).to(self.device)
                    train_loss_regularization.append(loss_reg.item())
                    if self.args.model == "CodimNet":
                        loss_total = loss_ce + nd_loss
                    else:
                        loss_total = loss_ce
                train_loss_total.append(loss_total.item())

                preds.append(logits.detach())
                trues.append(label_y)

                loss_total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                pbar.set_postfix({
                    "total": f"{loss_ce.item():.4f}",
                    "ce": f"{loss_ce.item():.4f}",
                    "regularization": f"{loss_reg.item():.4f}"
                })

            self.swa_model.update_parameters(self.model)

            infoTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\t')
            info = infoTime + "Itr: {} Epoch: {} cost time: {}".format(int(config.itr.split('_', 1)[1]), epoch + 1, time.time() - epoch_time)
            print(info)
            log_write.write(info + "\n")

            train_loss_instance = np.average(train_loss_instance)
            train_loss_subject = np.average(train_loss_subject)
            train_loss_ce = np.average(train_loss_ce)
            train_loss_regularization = np.average(train_loss_regularization)
            train_loss_total = np.average(train_loss_total)
            trues = torch.cat(trues, 0).flatten().cpu().numpy()
            predictions = torch.argmax(torch.nn.functional.softmax(torch.cat(preds, 0)), dim=-1).cpu().numpy()
            train_acc = accuracy_score(trues, predictions)
            train_F1 = f1_score(trues, predictions, average="macro")

            # 验证模型
            vali_loss, sample_val_metrics_dict, subject_val_metrics_dict = self.evaluation(vali_loader, criterion, eval_model='validation')
            _, sample_test_metrics_dict, subject_test_metrics_dict = self.evaluation(test_loader, criterion, eval_model='test')

            infoTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\n')
            info = (infoTime + 
                    f"========================== Train data results: ==========================\n" +
                    f"Train Loss Total: {train_loss_total:.5f}, Train Loss CE: {train_loss_ce:.5f}, Train Loss Regularization: {train_loss_regularization:.5f}, Train Loss Instance: {train_loss_instance:.5f}, Train Loss Subject: {train_loss_subject:.5f} \n" +
                    f"Train F1: {train_F1:.5f}, Train Acc: {train_acc:.5f}\n" +
                    f"========================== Validation data results: ==========================\n" +
                    f"========================== Sample-level results: \n" +
                    f"Val Loss Total: {vali_loss['total']:.5f}, Val Loss CE: {vali_loss['ce']:.5f}, Val Loss Regularization: {vali_loss['regularization']:.5f}, Val Loss Instance: {vali_loss['instance']:.5f}, Val Loss Subject: {vali_loss['subject']:.5f}\n" +
                    f"Accuracy: {sample_val_metrics_dict['Accuracy']:.5f}, " +
                    f"F1: {sample_val_metrics_dict['F1']:.5f}, " +
                    f"Recall: {sample_val_metrics_dict['Recall']:.5f}, " +
                    f"Precision: {sample_val_metrics_dict['Precision']:.5f}, " +
                    f"k-score: {sample_val_metrics_dict['k-score']:.5f}, " +
                    f"AUROC: {sample_val_metrics_dict['AUROC']:.5f}, " +
                    f"AUPRC: {sample_val_metrics_dict['AUPRC']:.5f}\n" +
                    f"Confusion Matrix: \n{sample_val_metrics_dict['Confusion Matrix']}\n" +
                    f"========================== Subject-level results: \n" +
                    f"Accuracy: {subject_val_metrics_dict['Accuracy']:.5f}, " +
                    f"F1: {subject_val_metrics_dict['F1']:.5f}, " +
                    f"Recall: {subject_val_metrics_dict['Recall']:.5f}, " +
                    f"Precision: {subject_val_metrics_dict['Precision']:.5f}, " +
                    f"k-score: {subject_val_metrics_dict['k-score']:.5f}, " +
                    f"AUROC: {subject_val_metrics_dict['AUROC']:.5f}, " +
                    f"AUPRC: {subject_val_metrics_dict['AUPRC']:.5f}\n" +
                    f"Confusion Matrix: \n{subject_val_metrics_dict['Confusion Matrix']}\n" +
                    f"========================== Test data results: ==========================\n" +
                    f"========================== Sample-level results: \n" +
                    f"Accuracy: {sample_test_metrics_dict['Accuracy']:.5f}, " +
                    f"F1: {sample_test_metrics_dict['F1']:.5f}, " +
                    f"Recall: {sample_test_metrics_dict['Recall']:.5f}, " +
                    f"Precision: {sample_test_metrics_dict['Precision']:.5f}, " +
                    f"k-score: {sample_test_metrics_dict['k-score']:.5f}, " +
                    f"AUROC: {sample_test_metrics_dict['AUROC']:.5f}, " +
                    f"AUPRC: {sample_test_metrics_dict['AUPRC']:.5f}, \n" +
                    f"Confusion Matrix: \n{sample_test_metrics_dict['Confusion Matrix']}\n" +
                    f"========================== Subject-level results: \n" +
                    f"Accuracy: {subject_test_metrics_dict['Accuracy']:.5f}, " +
                    f"F1: {subject_test_metrics_dict['F1']:.5f}, " +
                    f"Recall: {subject_test_metrics_dict['Recall']:.5f}, " +
                    f"Precision: {subject_test_metrics_dict['Precision']:.5f}, " +
                    f"k-score: {subject_test_metrics_dict['k-score']:.5f}, " +
                    f"AUROC: {subject_test_metrics_dict['AUROC']:.5f}, " +
                    f"AUPRC: {subject_test_metrics_dict['AUPRC']:.5f}\n" +
                    f"Confusion Matrix: \n{subject_test_metrics_dict['Confusion Matrix']}\n")
            print(info)
            log_write.write(info)

            # if config.visualization_training == 'Control' or config.visualization_training == 'Experimental':
            #     wandb.log({'epoch': epoch,
            #                'train_loss_total': train_loss_total, 'train_loss_ce': train_loss_ce,
            #                'train_acc': train_acc, 'train_F1': train_F1,
            #                'vali_loss_total': vali_loss['total'], 'vali_loss_ce': vali_loss['ce'],
            #                'val_Acc': sample_val_metrics_dict['Accuracy'], 'val_F1': sample_val_metrics_dict['F1'],
            #                'val_Recall': sample_val_metrics_dict['Recall'], 'val_Precision': sample_val_metrics_dict['Precision'], 'val_k-score': sample_val_metrics_dict['k-score'], 'val_AUROC': sample_val_metrics_dict['AUROC'], 'val_AUPRC': sample_val_metrics_dict['AUPRC'],
            #                'val_Confusion Matrix': sample_val_metrics_dict['Confusion Matrix'],
            #                'test_Acc': sample_test_metrics_dict['Accuracy'], 'test_F1': sample_test_metrics_dict['F1'],
            #                'test_Recall': sample_test_metrics_dict['Recall'], 'test_Precision': sample_test_metrics_dict['Precision'], 'test_k-score': sample_test_metrics_dict['k-score'], 'test_AUROC': sample_test_metrics_dict['AUROC'], 'test_AUPRC': sample_test_metrics_dict['AUPRC'],
            #                'test_Confusion Matrix': sample_test_metrics_dict['Confusion Matrix']})
            # elif config.visualization_training == 'Parameter_Adjustment':
            #     vis.line(X=[epoch], Y=[train_loss_total],                    win=train_loss_total_window,    opts=opt_train_loss_total,    update='append')
            #     vis.line(X=[epoch], Y=[train_loss_ce],                       win=train_loss_ce_window,       opts=opt_train_loss_ce,       update='append')
            #     vis.line(X=[epoch], Y=[train_loss_regularization],           win=train_loss_regularization_window, opts=opt_train_loss_regularization, update='append')
            #     vis.line(X=[epoch], Y=[train_acc],                           win=train_acc_window,           opts=opt_train_acc,           update='append')
            #     vis.line(X=[epoch], Y=[train_F1],                            win=train_F1_window,            opts=opt_train_F1,            update='append')
            #     vis.line(X=[epoch], Y=[vali_loss['total']],                  win=eval_loss_total_window,     opts=opt_eval_loss_total,     update='append')
            #     vis.line(X=[epoch], Y=[vali_loss['ce']],                     win=eval_loss_ce_window,        opts=opt_eval_loss_ce,        update='append')
            #     vis.line(X=[epoch], Y=[vali_loss['regularization']],         win=eval_loss_regularization_window, opts=opt_eval_loss_regularization, update='append')
            #     vis.line(X=[epoch], Y=[sample_val_metrics_dict['Accuracy']], win=eval_acc_window,            opts=opt_eval_acc,            update='append')
            #     vis.line(X=[epoch], Y=[sample_val_metrics_dict['F1']],       win=eval_F1_window,             opts=opt_eval_F1,             update='append')

            early_stopping(
                -sample_val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                checkpointsPath
            )
            if early_stopping.counter >= self.args.warmup_epochs and self.args.warmup_epochs > 0 and self.add_regularization == False:
                self.add_regularization = True
                early_stopping.counter = 0
            if early_stopping.early_stop:
                infoTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\t')
                info = infoTime + f'Earyly stopping at Epoch {epoch + 1},and retrain the Net using both the training data and validation data.\n'
                print(info + "\n")
                log_write.write(info + "\n")
                break

            # 更新验证集最优值
            if early_stopping.save_model:
                best_val['f1'] = sample_val_metrics_dict['F1']
                best_val['acc'] = sample_val_metrics_dict['Accuracy']
                best_val['to_test_f1'] = sample_test_metrics_dict['F1']
                best_val['to_test_acc'] = sample_test_metrics_dict['Accuracy']
                best_val['to_test_acc_sub'] = subject_test_metrics_dict['Accuracy']
            if sample_test_metrics_dict['F1'] > best_val['test_f1']:
                best_val['test_f1'] = sample_test_metrics_dict['F1']
                best_val['test_acc'] = sample_test_metrics_dict['Accuracy']
                best_val['test_acc_sub'] = subject_test_metrics_dict['Accuracy']
            info = (f"========================== Best Val results: ==========================\n" +
                    f"Best Val F1: {best_val['f1']:.5f}, Best Val Acc: {best_val['acc']:.5f}\n" +
                    f"Best To Test F1: {best_val['to_test_f1']:.5f}, Best To Test Acc: {best_val['to_test_acc']:.5f}, Best To Test Acc Sub: {best_val['to_test_acc_sub']:.5f}\n" +
                    f"Best Test F1: {best_val['test_f1']:.5f}, Best Test Acc: {best_val['test_acc']:.5f}, Best Test Acc Sub: {best_val['test_acc_sub']:.5f}\n")
            print(info)
            log_write.write(info)

            if scheduler:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()  # 调整学习率
            else:
                current_lr = model_optim.state_dict()['param_groups'][0]['lr']  # 当前学习率
            # vis.line(X=[epoch], Y=[float(current_lr)], win=lr_window, opts=opt_lr, update='append')

        if self.args.secondStep:
            best_model_path = checkpointsPath + "checkpoint.pth"
            self._load_checkpoint(best_model_path)

            for epoch in range(self.args.second_epochs):
                preds = []
                trues = []
                second_loss = []

                self.model.train()
                epoch_time = time.time()

                for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long())
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()
                    second_loss.append(loss.item())
                    preds.append(outputs.detach())
                    trues.append(label)

                for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long())
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()
                    second_loss.append(loss.item())
                    preds.append(outputs.detach())
                    trues.append(label)

                self.swa_model.update_parameters(self.model)

                infoTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\t')
                info = infoTime + "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
                print(info)
                log_write.write(info + "\n")

                second_loss = np.average(second_loss)
                trues = torch.cat(trues, 0)
                trues = trues.flatten().cpu().numpy()
                preds = torch.cat(preds, 0)
                probs = torch.nn.functional.softmax(preds, dim=-1)
                predictions = (torch.argmax(probs, dim=1).cpu().numpy())
                second_acc = accuracy_score(trues, predictions)

                infoTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\t')
                info = infoTime + f"Second Loss: {second_loss:.5f}, Second Acc: {second_acc:.5f}"
                print(info)
                log_write.write(info + "\n")

                early_stopping(
                    second_loss,
                    self.swa_model if self.swa else self.model,
                    checkpointsPath,
                )
                if early_stopping.early_stop:
                    infoTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S\t')
                    info = infoTime + f'Earyly stopping at Epoch {epoch + 1}, the second training step is end.'
                    print(info)
                    log_write.write(info + "\n")
                    break
                """if (epoch + 1) % 5 == 0:
                    adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        log_write.close()
        wandb.finish()

    #%%
    def test(self, config):
        checkpointsPath = (self.args.checkpoints + self.args.project_name + "/" + config.itr + "/")
        model_path = checkpointsPath + "checkpoint.pth"
        if not os.path.exists(model_path):
            raise Exception("No model found at %s" % model_path)
        self._load_checkpoint(model_path)

        test_data, test_loader = self._get_data(flag="TEST")

        criterion = self._select_criterion(labelNum=torch.bincount(torch.from_numpy(test_data.y[:, 0]).long()))
        _, sample_test_metrics_dict, subject_test_metrics_dict = self.evaluation(test_loader, criterion, eval_model='test')

        if self.args.is_training == 1:
            info = (f"========================== Test data results: ==========================\n" +
                    f"========================== Sample-level results: \n" +
                    f"Accuracy: {sample_test_metrics_dict['Accuracy']:.5f}, " +
                    f"F1: {sample_test_metrics_dict['F1']:.5f}, " +
                    f"Recall: {sample_test_metrics_dict['Recall']:.5f}, " +
                    f"Precision: {sample_test_metrics_dict['Precision']:.5f}, " +
                    f"k-score: {sample_test_metrics_dict['k-score']:.5f}, " +
                    f"AUROC: {sample_test_metrics_dict['AUROC']:.5f}, " +
                    f"AUPRC: {sample_test_metrics_dict['AUPRC']:.5f}, \n" +
                    f"Confusion Matrix: \n{sample_test_metrics_dict['Confusion Matrix']}\n" +
                    f"========================== Subject-level results: \n" +
                    f"Accuracy: {subject_test_metrics_dict['Accuracy']:.5f}, " +
                    f"F1: {subject_test_metrics_dict['F1']:.5f}, " +
                    f"Recall: {subject_test_metrics_dict['Recall']:.5f}, " +
                    f"Precision: {subject_test_metrics_dict['Precision']:.5f}, " +
                    f"k-score: {subject_test_metrics_dict['k-score']:.5f}, " +
                    f"AUROC: {subject_test_metrics_dict['AUROC']:.5f}, " +
                    f"AUPRC: {subject_test_metrics_dict['AUPRC']:.5f}\n" +
                    f"Confusion Matrix: \n{subject_test_metrics_dict['Confusion Matrix']}\n")
            print(info)

        return sample_test_metrics_dict, subject_test_metrics_dict
