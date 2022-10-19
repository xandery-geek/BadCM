from pytorch_lightning import callbacks


class MyProgressBar(callbacks.ProgressBarBase):
    def __init__(self) -> None:
        super().__init__()
        self.enable = True

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

