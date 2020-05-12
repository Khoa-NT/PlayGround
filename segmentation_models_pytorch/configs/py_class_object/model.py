class Merger:
    def __init__(self, cfg):
        self.cfg = cfg

    def shape(self):
        print(f"{type(self.cfg)}")
        print(f"cfg: {self.cfg.pretty()}\n")
        print(f"cfg: {self.cfg.exp_name}\n")

