import numpy as np
from tensorboardX import SummaryWriter

from datetime import datetime
import os

class Logger(SummaryWriter):
    def __init__(self, logdir):
        hist = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logger_dir = os.path.join(logdir, hist)
        super(Logger, self).__init__(self.logger_dir)

    def write_train(self, training_loss, iteration):
        self.add_scalar('training loss', training_loss, iteration)

    def write_val(self, validation_loss, audio, sample_rate, iteration):
        self.add_scalar('validation loss', validation_loss, iteration)
        self.add_audio('sample', audio, iteration, sample_rate)

        np.save(self.logger_dir + '/sample{}'.format(iteration), audio)
