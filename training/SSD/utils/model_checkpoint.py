import os
import pickle
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback


class ModelCheckpoint(Callback):
    """ A checkpoint to save a model checkpoint every n batches (iterations) or n epochs.

    Args:
        - output_dir: Path to output directory in which to save the checkpoint.
    """

    def __init__(self, output_dir, epoch_frequency, iteration_frequency, initial_epoch):
        self.output_dir = output_dir
        self.iteration_frequency = iteration_frequency
        self.epoch_frequency = epoch_frequency
        self.initial_epoch = initial_epoch
        self.iterations = 1
        self.epochs = 1
        self.losses_by_iteration = []
        hardcoded_losses = [128,11.7661,8.8221,8.2688,7.9703,7.7830,7.4332,7.2364,7.1183,7.0293,6.9677,6.8250]
        self.plotting_losses_idx = [10,20,30,40,50,60,70,80,90,100,110]
        self.losses_by_epoch = [] #[int(num) for num in hardcoded_losses]

    def on_batch_end(self, batch, logs={}):
        if self.iteration_frequency is not None:
            loss = logs["loss"]
            self.losses_by_iteration.append(loss)
            if self.iterations % self.iteration_frequency == 0:
                loss = '%.4f' % loss
                name = f"cp_it_{self.iterations}_loss_{loss}.h5"
                self.model.save_weights(os.path.join(self.output_dir, name))
                plt.plot(list(range(1, self.iterations+1)), self.losses_by_iteration)
                plt.title('training loss')
                plt.ylabel('loss')
                plt.xlabel('iteration')
                plt.savefig(os.path.join(self.output_dir, "log.png"))
        self.iterations += 1

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch_frequency is not None:
            loss = logs["loss"]
            self.losses_by_epoch.append(loss)
            loss_file = 'losses_by_epoch.pickle'
            with open(os.path.join(self.output_dir, loss_file), 'wb') as handle:
                pickle.dump(self.losses_by_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if self.epochs % self.epoch_frequency == 0:
                loss = '%.4f' % loss
                name = f"cp_ep_{self.epochs+self.initial_epoch}_loss_{loss}.h5"
                self.model.save_weights(os.path.join(self.output_dir, name))
                # plt.plot(list(range(1, self.epochs+1)), self.losses_by_epoch)
                # plt.title('training loss')
                # plt.ylabel('loss')
                # plt.xlabel('epoch')
                # plt.savefig(os.path.join(self.output_dir, "log.png"))
        self.epochs += 1
