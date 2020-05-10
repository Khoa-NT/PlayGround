class Merger:
    def __init__(self, layer, channel):
        self.layer = layer
        self.channel = channel

    def shape(self):
        print(f"model_1: {self.__class__.__name__}\n"
              f"layer: {self.layer}\n"
              f"channel: {self.channel}\n")
        pass
