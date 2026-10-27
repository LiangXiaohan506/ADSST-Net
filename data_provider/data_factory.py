from torch.utils.data import DataLoader
from data_provider.uea import collate_fn
from data_provider.uea import CustomGroupSampler

from data_provider.dataset_loader.adftd_loader import ADFTDLoader
from data_provider.dataset_loader.caeeg_loader import CAEEGLoader
from data_provider.dataset_loader.caueeg_loader import CAUEEGLoader
from data_provider.dataset_loader.iumsmci_loader import IUMSMCILoader



data_dict = {
    "CAEEG": CAEEGLoader,  # dataset CAEEG
    "ADFTD": ADFTDLoader,  # dataset ADFTD
    "CAUEEG": CAUEEGLoader,  # dataset CAUEEG
    "IUMSMCI": IUMSMCILoader,  # dataset IUMSMCI
}


def data_provider(args, flag, log):          
    data_set = data_dict[args.data](args=args, flag=flag, log=log) # 数据已经被洗牌

    sampler = CustomGroupSampler(args, data_set, batch_size=args.batch_size, group_size=args.group_size)
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # only called when yielding batches
    )

    return data_set, data_loader