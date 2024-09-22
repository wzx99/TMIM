from datasets.dataset_tmim import build_dataset as build_tmim_dataset
from datasets.dataset_str import build_dataset as build_str_dataset

def build_dataset(cfg):
    if cfg.data_type=='str':
        return build_str_dataset(cfg)
    elif cfg.data_type=='tmim':
        return build_tmim_dataset(cfg)