"""
Data loader for EFRing data

ver. 1.4.8

Created by taizhou 03/09/2021

Update Log:
1.4.9 13/01/2022    Use new approach for calculating channel wise diff by taking|   taizhou
		    considiation of channel distance
1.4.8 22/12/2021    Modify time_to_freq function to support one sample as input |   taizhou
                    Set random seed to 42 for splitting training set and test set
                    Modify _prepare_temporal_data function in DataLoaderContinuous 
                    to form temporal sliding window feature
1.4.7 11/12/2021    Update DataLoaderContinuous class                           |   taizhou
1.4.6 20/11/2021    Increase loading speed for testing                          |   taizhou
                    Change the output of time_to_freq to channel-first order
1.4.5 20/11/2021    Add function _channel_wise_diff                             |   taizhou
                    Add string augment "feature" to select specific features
1.4.4 19/11/2021    Add output_size augment for function time_to_freq           |   taizhou
1.4.3 09/11/2021    Change the logic for searching data files:                  |   taizhou
                        - files was searched recursively, maximum depth 2, 
                          the hierarchy should look like [path]/[userID]/[files]
                        - every individual file name under the "path" should
                          be unique. If repeated named found, use the first 
                          file as default
1.4.2 04/11/2021    Add time_to_freq function to get the stacked frequency map  |   taizhou
1.4.1 02/11/2021    Treat data with length larger than 210 as invalided data    |   taizhou 
1.4.0 04/10/2021    Add a DataLoaderContinuous class for regression             |   taizhou
1.3.1 20/09/2021    Only do data augmentation while training.                   |   taizhou
1.3.0 19/09/2021    Support data augmentation by time domain shifting           |   taizhou
                    Apply low pass filter for pre-process
1.2.1 17/09/2021    Support zero centering and l1 norm.                         |   taizhou
                    Fix bug while nb_window is 1
1.1.1 07/09/2021    Support for assigning training files                        |   taizhou
1.0.1 06/09/2021    Fix bug                                                     |   taizhou

"""

import numpy as np
import glob
import os
import random
from numpy.core.defchararray import find
from scipy import signal
from numpy.lib.function_base import place
from numpy.lib.polynomial import poly, polyfit
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

def time_to_freq(X, backend="scipy", output_size=224, window_size=10, step=5, n_fft=2048):

    if backend == "tf":
        import tensorflow as tf

    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=0)
    
    import cv2
    new_X = np.zeros((X.shape[0], X.shape[-1], output_size, output_size))
    for i, x in enumerate(X):

        new_x = np.zeros((x.shape[-1], output_size, output_size))
        for c in range(x.shape[-1]):

            if backend == "tf":
                spectrogram = tf.signal.stft(
                    x[:, c], frame_length=window_size, frame_step=step, fft_length=n_fft)
                spectrogram = tf.abs(spectrogram)[0:spectrogram.shape[0]//4]
                spectrogram_np = np.flip(tf.transpose(spectrogram).numpy(), axis=0)
            else:
                _, _, spectrogram = signal.stft(x[:, c], 190, nperseg=window_size, noverlap=window_size-step, nfft=n_fft)
                spectrogram = spectrogram.astype(np.float64)[0:spectrogram.shape[0]//4]
                spectrogram_np = np.flip(spectrogram, axis=0)

            spectrogram_np = (spectrogram_np - np.min(spectrogram_np)) / (np.max(spectrogram_np) - np.min(spectrogram_np))
            mat = cv2.resize(spectrogram_np, (output_size, output_size))

            
            new_x[c, ...] = mat 

        # for c in range(x.shape[-1]):
        #     cv2.imshow(str(c), new_x[..., c])
        #     cv2.waitKey()

        new_X[i] = new_x

    return new_X

class DataLoader(object):

    def __init__(self, 
                 path: str, # path to data folder
                 bath_size = None, # NO USE
                 window_size: int = 80, 
                 nb_windows: int = 3,
                 offset: int = 5, # offset for removing head and tail
                 max_len: int = 190, # maximun length of sequence
                 aug_rate: int = 0, # augmentation rate, between 0 and 1
                 cutoff_freq: int = 80, # the cutoff frequency while applying low pass filter
                 feature: str = "sd", # feature to use, "sd", "cic", or "all"
                 train_file: list = [], # files for training. If it is empty, then use all files for training
                 val_file: list = [], # files for validating. If it is empty, then use 1/3 of training data for validating
                 test_file: list = []
                 ) -> None:
        
        self._path = path
        self._batch_size = bath_size
        self._window_size = window_size
        self._nb_windows = nb_windows
        self._offset = offset
        self._max_len = max_len
        self._cutoff_freq = cutoff_freq
        self._aug_rate = aug_rate
        self._feature = feature.lower()

        if not isinstance(train_file, list):
            train_file = [train_file]
        if not isinstance(val_file, list):
            val_file = [val_file]
        if not isinstance(test_file, list):
            test_file = [test_file]
        
        self._all_file = self._get_all_file()
        
        self._all_file = [os.path.basename(a) for a in self._all_file]
        self._train_file = [os.path.basename(a) for a in train_file]
        self._val_file = [os.path.basename(a) for a in val_file] 
        self._test_file = [os.path.basename(a) for a in test_file]

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        # if no testing
        if len(self._test_file) == 0: 
        
            # if no train file assigned, use all file for training
            if len(self._train_file) == 0:
                self._train_file = self._all_file
            
            if len(self._val_file) != 0:

                # remove duplicated files from the training file list
                self._train_file = [f for f in self._train_file if f not in self._val_file]

                _train_data_all = []
                _train_labels_all = []
                _val_data_all = []
                _val_labels_all = []

                for train_file_ in self._train_file:
                    data, labels = self._get_all_data_from_file(train_file_, self._pre_process, True)
                    _train_data_all += data
                    _train_labels_all += labels

                for val_file_ in self._val_file:
                    data, labels = self._get_all_data_from_file(val_file_, self._pre_process, False)
                    _val_data_all += data
                    _val_labels_all += labels

                if len(_train_data_all) == 0 or len(_train_labels_all) == 0 \
                    or len(_val_data_all) == 0 or len(_val_labels_all) == 0:
                    print('Cannot find any valid data')
                    exit()

                self.X_train = np.asarray(_train_data_all)
                self.y_train = np.asarray(_train_labels_all)
                self.X_val = np.asarray(_val_data_all)
                self.y_val = np.asarray(_val_labels_all)

            # if no validation file assigned, use 1/3 of training data for validating
            else:
                _all_data = []
                _all_labels = []
                for file in self._train_file:
                    data, labels = self._get_all_data_from_file(file, self._pre_process, True)
                    _all_data += data
                    _all_labels += labels

                if len(_all_data) == 0 or len(_all_labels) == 0:
                    print('Cannot find any valid data')
                    exit()

                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(_all_data, _all_labels, test_size=0.33, random_state=42)

                self.X_train = np.asarray(self.X_train)
                self.y_train = np.asarray(self.y_train)
                self.X_val = np.asarray(self.X_val)
                self.y_val = np.asarray(self.y_val) 

    def _get_all_file(self, suffix='.csv'):
        return glob.glob(os.path.join(self._path, '**/*'+suffix))

    def _get_all_data_from_file(self, filename: str, callback_func, is_training):

        files_ = self._get_all_file()
        filename_ = [f for f in files_ if f.find(filename) != -1]

        with open(filename_[0]) as f:
            lines = f.readlines()

        data = []
        labels = []
        samples = []
        for line in lines:
            if line[0] == '\t':

                if len(samples) != 0:
                    if len(samples) < 150 or len(samples) > 210: # abnormal data
                        labels.pop()
                    else:

                        data.append(callback_func(np.asarray(samples), is_training)) # data for current sample

                        ### remove labels start ###
                        #if labels[-1] == 6:
                        #    labels.pop()
                        #    data.pop()
                        #elif labels[-1] > 6:
                        #    labels[-1] = labels[-1] - 1
                        ### remove labels end ###

                labels.append(int(line.split(',')[3])) # label for next sample
                
                samples = []
            else:
                samples.append(np.fromstring(line, sep=','))

        if len(samples) != 0:
            if len(samples) < 150 or len(samples) > 210: # abnormal data
                labels.pop()
            else:
                data.append(callback_func(np.asarray(samples), is_training)) # data for last sample

                ### remove labels start ###
                #if labels[-1] == 6:
                #    labels.pop()
                #    data.pop()
                #elif labels[-1] > 6:
                #    labels[-1] = labels[-1] - 1
                ### remove labels end ###

        return data, labels

    def _pre_process(self, data, is_training):

        if self._feature == "sd":
            data = data[:, 0:5]
        if self._feature == "cic":
            data = data[:, 5:10]
        
        # offseting
        data = data[self._offset:-self._offset]

        # resampling
        data = self._resample(data, num=self._max_len)

        # zero centering
        # data = self._zero_centered(data)

        # low pass filtering
        data_filtered = np.ones_like(data)
        for i in range(data.shape[1]):
            data_filtered[:, i] = self._butter_lowpass_filter(data[:, i], sample_rate=190, cutoff=self._cutoff_freq)
        data = data_filtered

        # first order derivative along time
        data_dtime = np.gradient(data, axis=0, edge_order=1)

        # first order derivative between channel
        data_dchannel = self._channel_wise_diff(data)

        # concating features
        data = np.concatenate((data_dtime, data_dchannel), axis=-1)

        # augmentation
        if is_training and random.random() < self._aug_rate:
            data = self._augmentation(data)

        # normalizing
        data = self._normolize(data)

        # slicing
        data = self._prepare_sliding_window(data, window_size=self._window_size, nb_windows=self._nb_windows)
        
        return data

    def _channel_wise_diff(self, data):
        """
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
        """

        data_1_index = []
        data_2_index = []

        for i in range(data.shape[1]):
            for j in range(data.shape[1] - i - 1):
                data_1_index.append(i)
                data_2_index.append(j+1+i)

        data_1 = data[:, data_1_index]
        data_2 = data[:, data_2_index]

        data = data_1 - data_2
        data = data/np.abs(np.array(data_1_index) - np.array(data_2_index))
        
        return data

    def _zero_centered(self, x):
        for c in range(x.shape[1]):
            x[:, c] -= np.mean(x[:, c])

        return x

    def _resample(self, data, num=190):
        """
        upsample by repeating the last element

        """

        if data.shape[0] == num:
            return data
        elif data.shape[0] > num:
            data = data[:num, :]
        else:
            data = np.concatenate((data, np.tile(data[-1, :], (num-data.shape[0], 1))))

        return data

    def _normolize(self, data):

        max=np.max(data,axis=0)
        min=np.min(data,axis=0)

        data = (data - min) / (max - min)

        return data

    def _l1_norm(self, data):
        return data / np.linalg.norm(data)

    def _prepare_sliding_window(self, data, window_size, nb_windows):
        """
        slice the data into window pieces

        """

        if window_size > data.shape[0] or window_size == 0 or nb_windows == 0:
            return data

        if nb_windows == 1:
            return np.expand_dims(data, 0)

        step = (data.shape[0] - window_size) // (nb_windows - 1)

        output_data = []

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        for window_pointer in range(0, data.shape[0] - window_size + 1, step):
            output_data.append(data[window_pointer:window_pointer + window_size, :])

        output_data = np.asarray(output_data)

        # print("Data was converted to fit silding windows model from {} to {}".format(data.shape, output_data.shape))

        return output_data

    def _roll(self, data, offset):

        data = np.roll(data, offset, axis=0)

        all_zero = np.zeros((offset, data.shape[1]))
        data[0:offset, :] = all_zero

        return data


    def _augmentation(self, data):

        # offset the signal in time domain
        _offset = random.randint(1, data.shape[0] // 4)
        data = self._roll(data, _offset)
    
        return data

    def _butter_highpass_filter(self, data, sample_rate, cutoff, order=2):

        nyq = 0.5 * sample_rate  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    def _butter_lowpass_filter(self, data, sample_rate, cutoff, order=2):
        nyq = 0.5 * sample_rate  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def get_all_data(self):
        """
        get all data with label
        
        """
        all_data = []
        all_label = []
        for file in self._all_file:
            data, label = self._get_all_data_from_file(file, self._pre_process, False)
            all_data += data
            all_label += label

        return np.asarray(all_data), np.asarray(all_label)

    def get_test_data(self):
        """
        get test data with label

        """
        
        if len(self._test_file) == 0:
            print('No files for testing')
            exit()

        test_data = []
        test_label = []
        for file in self._test_file:
            data, label = self._get_all_data_from_file(file, self._pre_process, False)
            test_data += data
            test_label += label

        self.X_test = np.asarray(test_data)
        self.y_test = np.asarray(test_label)

        return self.X_test, self.y_test

    def get_data(self):
        """
        get data for training and validating

        """

        return self.X_train, self.X_val, self.y_train, self.y_val

    def get_nb_class(self):
        """
        get number of class

        """

        if self.y_train is not None and self.y_train.shape[0] > 0:
            return np.unique(self.y_train).shape[0]

        if self.y_test is not None and self.y_test.shape[0] > 0:
            return np.unique(self.y_test).shape[0]

class DataLoaderContinuous(object):

    def __init__(self, 
                 path: str, # path to data folder
                 bath_size = None, # NO USE
                 window_size: int = 3, 
                 offset: int = 5, # offset for removing head and tail
                 aug_rate: int = 0, # NO USE, augmentation rate, between 0 and 1
                 cutoff_freq: int = 90, # the cutoff frequency while applying low pass filter
                 feature: str = "sd", # feature to use, "sd", "cic", or "all"
                 train_file: list = [], # files for training. If it is empty, then use all files for training
                 val_file: list = [], # files for validating. If it is empty, then use 1/3 of training data for validating
                 test_file: list = []
                 ) -> None:
        
        self._path = path
        self._batch_size = bath_size
        self._window_size = window_size
        self._offset = offset
        self._cutoff_freq = cutoff_freq
        self._aug_rate = aug_rate
        self._feature = feature.lower()

        if not isinstance(train_file, list):
            train_file = [train_file]
        if not isinstance(val_file, list):
            val_file = [val_file]
        if not isinstance(test_file, list):
            test_file = [test_file]
        
        self._all_file = self._get_all_file()
        self._all_file = [os.path.basename(a) for a in self._all_file]
        self._train_file = [os.path.basename(a) for a in train_file]
        self._val_file = [os.path.basename(a) for a in val_file] 
        self._test_file = [os.path.basename(a) for a in test_file]   

        # if no train file assigned, use all file for training
        if len(self._train_file) == 0:
            self._train_file = self._all_file
        
        if len(self._val_file) != 0:

            # remove duplicated files from the training file list
            self._train_file = [f for f in self._train_file if f not in self._val_file]

            _train_data_all = []
            _train_labels_all = []
            _val_data_all = []
            _val_labels_all = []

            for train_file_ in self._train_file:
                data, labels = self._get_all_data_from_file(
                    train_file_, 
                    data_callback_func = self._pre_process_data, 
                    label_callback_func = self._pre_process_lable,
                    is_training = True
                    )
                _train_data_all += data
                _train_labels_all += labels

            for val_file_ in self._val_file:
                data, labels = self._get_all_data_from_file(
                    val_file_, 
                    data_callback_func = self._pre_process_data, 
                    label_callback_func = self._pre_process_lable,
                    is_training = True
                    )
                _val_data_all += data
                _val_labels_all += labels

            if len(_train_data_all) == 0 or len(_train_labels_all) == 0 \
                or len(_val_data_all) == 0 or len(_val_labels_all) == 0:
                print('Cannot find any valid data')
                exit()

            self.X_train, self.y_train = self._prepare_temporal_data(_train_data_all, _train_labels_all)
            self.X_val, self.y_val = self._prepare_temporal_data(_val_data_all, _train_labels_all)

        # if no validation file assigned, use 1/3 of training data for validating
        else:
            _all_data = []
            _all_labels = []
            for file in self._train_file:
                data, labels = self._get_all_data_from_file(
                    file, 
                    data_callback_func = self._pre_process_data, 
                    label_callback_func = self._pre_process_lable,
                    is_training = True
                    )
                _all_data += data
                _all_labels += labels

            if len(_all_data) == 0 or len(_all_labels) == 0:
                print('Cannot find any valid data')
                exit()

            _all_data_temporal, _all_labels_temporal = self._prepare_temporal_data(_all_data, _all_labels)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                _all_data_temporal, 
                _all_labels_temporal, 
                test_size=0.33,
                random_state=42
                )

            self.X_train = np.asarray(self.X_train)
            self.y_train = np.asarray(self.y_train)
            self.X_val = np.asarray(self.X_val)
            self.y_val = np.asarray(self.y_val) 

    def _get_all_file(self, suffix='.csv'):
        return glob.glob(os.path.join(self._path, '**/*'+suffix))

    def _get_all_data_from_file(self, filename: str, data_callback_func, label_callback_func, is_training):

        files_ = self._get_all_file()
        filename_ = [f for f in files_ if f.find(filename) != -1]

        with open(filename_[0]) as f:
            lines = f.readlines()

        samples = []
        y = []
        data = []
        labels = []
        for line in lines:
            if line[0] == '\t':
                if len(samples) != 0:
                    if len(samples) < 150: # abnormal data
                        pass
                    else:
                        data.append(data_callback_func(np.asarray(samples), is_training))
                        labels.append(label_callback_func(y))
                samples = []
                y = []
            else:
                lines_np = np.fromstring(line, sep=',')
                samples.append(lines_np[0:-1])
                y.append(lines_np[-1])

        if len(samples) != 0:
            if len(samples) < 150: # abnormal data
                pass
            else:
                data.append(data_callback_func(np.asarray(samples), is_training))
                labels.append(label_callback_func(y))

        return data, labels

    def _pre_process_data(self, data, is_training):

        if self._feature == "sd":
            data = data[:, 0:5]
        if self._feature == "cic":
            data = data[:, 5:10]
        
        # offseting
        # data = data[self._offset:-self._offset]

        # zero centering
        # data = self._zero_centered(data)

        # first order derivative
        # data = np.gradient(data, axis=0)
        # data = np.diff(data, axis=0)

        # low pass filtering
        # data = self._butter_lowpass_filter(data, sample_rate=200, cutoff=self._cutoff_freq)

        # normalizing
        # data = self._normolize(data)

        # slicing
        # data = self._prepare_sliding_window(data, window_size=self._window_size, nb_windows=self._nb_windows)
        
        return data

    def _pre_process_lable(self, label):

        # offseting
        # label = label[self._offset:-self._offset]

        # first order derivative
        # label = np.gradient(label, axis=0)
        # label = np.diff(label, axis=0)

        return label

    def _zero_centered(self, x):
        for c in range(x.shape[1]):
            x[:, c] -= np.mean(x[:, c])

        return x

    def _resample(self, data, num=190):
        """
        upsample by repeating the last element

        """

        if data.shape[0] == num:
            return data
        elif data.shape[0] > num:
            data = data[:num, :]
        else:
            data = np.concatenate((data, np.tile(data[-1, :], (num-data.shape[0], 1))))

        return data

    def _normolize(self, data):

        max=np.max(data,axis=0)
        min=np.min(data,axis=0)

        data = (data - min) / (max - min)

        return data

    def _l1_norm(self, data):
        return data / np.linalg.norm(data)

    def _prepare_sliding_window(self, data, window_size, nb_windows):
        """
        slice the data into window pieces

        """

        if window_size > data.shape[0] or window_size == 0 or nb_windows == 0:
            return data

        if nb_windows == 1:
            return np.expand_dims(data, 0)

        step = (data.shape[0] - window_size) // (nb_windows - 1)

        output_data = []

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        for window_pointer in range(0, data.shape[0] - window_size + 1, step):
            output_data.append(data[window_pointer:window_pointer + window_size, :])

        output_data = np.asarray(output_data)

        # print("Data was converted to fit silding windows model from {} to {}".format(data.shape, output_data.shape))

        return output_data

    def _prepare_temporal_data(self, data, label, window_size=3):
        """
        forming data into temporal format, data is suppose to be a (nb_sample, len_sample, nb_channel) tensor
        
        """
        output_data = []
        output_label = []
        if window_size is None:
            # without using sliding window
            for i, sample in enumerate(data):
                for j, frame in enumerate(sample):
                    output_data.append(frame)
                    output_label.append(label[i][j])
        else:
            for i, sample in enumerate(data):
                window = np.zeros((5 * window_size, ))
                for j, frame in enumerate(sample):
                    if j < window_size - 1:
                        window[j*5: j*5+5] = frame
                        
                    else:
                        window[(window_size-1) * 5: window_size * 5] = frame
                        
                        output_data.append(window)
                        output_label.append(label[i][j])
                        window = np.roll(window, -5)

                    
        return np.asarray(output_data), np.asarray(output_label)


    def _roll(self, data, offset):

        data = np.roll(data, offset, axis=0)

        return data


    def _augmentation(self, data):

        # offset the signal in time domain
        _offset = random.randint(1, data.shape[0] // 4)
        data = self._roll(data, _offset)

        return data

    def _butter_highpass_filter(self, data, sample_rate, cutoff, order=2):

        nyq = 0.5 * sample_rate  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    def _butter_lowpass_filter(self, data, sample_rate, cutoff, order=2):
        nyq = 0.5 * sample_rate  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def get_all_data(self):
        """
        get all data with label
        
        """
        all_data = []
        all_label = []
        for file in self._all_file:
            data, label = self._get_all_data_from_file(
                file, 
                data_callback_func = self._pre_process_data, 
                label_callback_func = self._pre_process_lable,
                is_training = False
                )
            all_data += data
            all_label += label

        return np.asarray(all_data), np.asarray(all_label)

    def get_test_data(self):
        """
        get test data with label

        """
        
        if len(self._test_file) == 0:
            print('No files for testing')
            exit()

        test_data = []
        test_label = []
        for file in self._test_file:
            data, label = self._get_all_data_from_file(
                file, 
                data_callback_func = self._pre_process_data, 
                label_callback_func = self._pre_process_lable,
                is_training = False
                )
            test_data += data
            test_label += label

        return self._prepare_temporal_data(test_data, test_label)

    def get_data(self):
        """
        get data for training and validating

        """

        return self.X_train, self.X_val, self.y_train, self.y_val


if __name__ == "__main__":

    mDataLoader = DataLoader('../gesture_selection/recording', 
        nb_windows=1, 
        aug_rate=0.5,
        train_file="gestic_data_off_taizhou_selection.csv")
    X_train, X_val, y_train, y_val = mDataLoader.get_data()

    print("{}, {}, {}, {}".format(X_train.shape, X_val.shape, y_train.shape, y_val.shape))

    print("{}, {}, {}, {}".format(time_to_freq(np.squeeze(X_train)).shape, time_to_freq(np.squeeze(X_val)).shape, y_train.shape, y_val.shape))

    # mDataLoader = DataLoaderContinuous('./from_time_domain/new_data/continuous', 
    #         train_file = 'gestic_data_user0_linearSlider.csv')
    # X_train, X_val, y_train, y_val = mDataLoader.get_data()

    # print("{}, {}, {}, {}".format(X_train.shape, X_val.shape, y_train.shape, y_val.shape))
