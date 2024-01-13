import os
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, logdir=""):
        self.logdir = logdir

        if not os.path.exists(self.logdir):
            print(f"Making logging dir '{self.logdir}'")
            os.makedirs(self.logdir)
        self.writer = SummaryWriter(self.logdir)

    def log_train(self, tag, v, i):
        self.writer.add_scalar(f"Train/{tag}", v, i)

    def log_train_fig(self, tag, fig, i):
        self.writer.add_figure(f"Train/{tag}", fig, i, close=True)

    def log_train_info(self, info_dict, i):
        for k in info_dict:
            self.writer.add_scalar(f"Train/{k}", info_dict[k], i)

    def log_test(self, tag, v, i):
        self.writer.add_scalar(f"Test/{tag}", v, i)

    def log_test_fig(self, tag, fig, i):
        self.writer.add_figure(f"Test/{tag}", fig, i, close=True)

    def log_test_info(self, info_dict, i):
        for k in info_dict:
            self.writer.add_scalar(f"Test/{k}", info_dict[k], i)

    def log_pretrain(self, tag, v, i):
        self.writer.add_scalar(f"Pre-Train/{tag}", v, i)

    def log_pretrain_fig(self, tag, fig, i):
        self.writer.add_figure(f"Pre-Train/{tag}", fig, i, close=True)

    def log_pretrain_info(self, info_dict, i):
        for k in info_dict:
            self.writer.add_scalar(f"Pre-Train/{k}", info_dict[k], i)

    def flush(self):
        self.writer.flush()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    logger = Logger("./log")
    plt.figure("123")
    for i in range(10):
        a = plt.figure("456")
        a.gca().plot(list(range(100)))
        logger.log_train_fig("cs", a, i)

    plt.plot(list(range(100, 0, -1)))
    plt.show()
