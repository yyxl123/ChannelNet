import logging
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy
from scipy.io import loadmat

from models import interpolation, SRCNN_train, SRCNN_predict

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)

def application():
    """ All application has its initialization from here """
    logging.info('Main application is running!')

    #channel_net_train()
    channel_net_predict()

class ElapsedTime(object):
    """
    Measure the elapsed time from "elapsed.tic()" to "elapsed.elapsed()"
    """

    def __init__(self):
        self.start_time = None

    def tic(self):
        self.start_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            logger.warning("ElapsedTime: tic() was not called before elapsed().")
            return
        _elapsed = time.time() - self.start_time
        d = timedelta(seconds=_elapsed)
        logger.info('< {} >'.format(d))

class ChannelInfo:
    """
    Channel constructor details
    """

    def __init__(self):
        """
        # load dataset
        """
        self.channel_model = "VehA"
        self.SNR = 12
        self.number_of_pilots = 48
        self.perfect = loadmat("Perfect_H_40000.mat")['My_perfect_H']
        self.noisy_input = loadmat("My_noisy_H_12.mat")["My_noisy_H"]

channel_info = ChannelInfo()

def channel_net_format_data():
    """
    Deep Learning
    """

    noisy_input = channel_info.noisy_input
    snr = channel_info.SNR
    perfect = channel_info.perfect
    number_of_pilots = channel_info.number_of_pilots
    channel_model = channel_info.channel_model

    interp_noisy = interpolation(noisy_input, snr, number_of_pilots, 'rbf')

    perfect_image = numpy.zeros((len(perfect), 72, 14, 2))
    perfect_image[:, :, :, 0] = numpy.real(perfect)
    perfect_image[:, :, :, 1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(
        2 * len(perfect), 72, 14, 1)

    idx_random = numpy.random.rand(len(perfect_image)) < (1 / 9)  # 32000 as training, the rest as validation
    train_data, train_label = interp_noisy[idx_random, :, :, :], perfect_image[idx_random, :, :, :]
    val_data, val_label = interp_noisy[~idx_random, :, :, :], perfect_image[~idx_random, :, :, :]

    return train_data, train_label, val_data, val_label

train_data, train_label, val_data, val_label = channel_net_format_data()

def channel_net_train():
    elapsed = ElapsedTime()
    elapsed.tic()

    #train_data, train_label, val_data, val_label = channel_net_format_data()
    SRCNN_train(train_data, train_label, val_data, val_label, channel_info.channel_model, channel_info.number_of_pilots, channel_info.SNR)

    elapsed.elapsed()

def calculate_mse(pred, label):
    """
    计算预测值和真实标签之间的均方误差（MSE）。
    pred 和 label 的形状为 (num_samples, 72, 14, 1)
    """
    # 将 pred 和 label 转换为 2D 数组（num_samples, 72*14）
    pred_flat = pred.reshape(pred.shape[0], -1)
    label_flat = label.reshape(label.shape[0], -1)

    # 计算每个样本的 MSE
    mse_per_sample = numpy.mean((pred_flat - label_flat) ** 2, axis=1)

    # 返回所有样本的平均 MSE
    return numpy.mean(mse_per_sample)

def channel_net_predict():
    elapsed = ElapsedTime()
    elapsed.tic()

    #train_data, train_label, val_data, val_label = channel_net_format_data()
    #print(train_data, train_label, val_data, val_label)

    # ------ prediction using SRCNN ------
    srcnn_pred = SRCNN_predict(train_data, channel_info.channel_model, channel_info.number_of_pilots, channel_info.SNR)
    #print(srcnn_pred.shape)
    # ------ 计算MSE ------
    mse = calculate_mse(srcnn_pred, train_label)

    print(mse)

    elapsed.elapsed()
    return None
