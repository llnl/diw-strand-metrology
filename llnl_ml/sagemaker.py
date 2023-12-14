from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


SM_METRICS_DEFINITIONS = [
    {"Name": "train:loss", "Regex": "train_loss=(.*?)[,\]]"},
    {"Name": "val:loss", "Regex": "val_loss=(.*?)[,\]]"},
    {"Name": "learning_rate", "Regex": "lr=(.*?)[,\]]"},
    {"Name": "val:acc", "Regex": "val_acc=(.*?)[,\]]"},
    {"Name": "val:jaccard", "Regex": "val_jaccard=(.*?)[,\]]"},
    {"Name": "val:dice", "Regex": "val_dice=(.*?)[,\]]"},
    {"Name": "test:loss", "Regex": "test_loss=(.*?)[,\]]"},
    {"Name": "test:acc", "Regex": "test_acc=(.*?)[,\]]"},
    {"Name": "test:jaccard", "Regex": "test_jaccard=(.*?)[,\]]"},
    {"Name": "test:dice", "Regex": "test_dice=(.*?)[,\]]"},
]


class SMExperimentsLogger(Logger):
    def __init__(self, run):
        super().__init__()
        self.run = run

    @property
    def name(self):
        return "SMExperimentsLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        self.run.log_parameters(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for name, value in metrics.items():
            self.run.log_metric(name=name, value=value, step=step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
