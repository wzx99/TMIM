from .networks.uformer import UformerTMIM,UformerSTR
from .networks.erasenet import ErasenetTMIM,ErasenetSTR
from .networks.pert import PertTMIM,PertSTR

def build_model(cfg):
    return eval(cfg.model_name)(**cfg)
