class SeedRecAgent():
    def __init__(self, cfg):
        self.cfg = cfg

    def connect(self) -> None:
        print(self.cfg.pretty())