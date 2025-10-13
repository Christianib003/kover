from dataclasses import dataclass
import yaml

@dataclass
class Cfg:
    d: dict
    def __getattr__(self, k): return self.d[k]

def load_config(path: str) -> Cfg:
    with open(path, "r") as f:
        return Cfg(yaml.safe_load(f))
