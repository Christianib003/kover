import yaml


class Cfg(dict):
    __getattr__ = dict.get
def load_config(path:str)->Cfg:
    with open(path,"r") as f: return Cfg(yaml.safe_load(f))
