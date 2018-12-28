
from __future__ import division
from __future__ import print_function

import json
import math
import multiprocessing as mp
import numpy as np
import os
import random
import librosa

class AudioProducer(object):

    def __init__(self, data_jsons, batch_size, sample_rate=8000,
                 min_duration=.3, max_duration=16.0):
        """
        Args:
            data_json : List of paths to files with speech data in json
                format. Each line should be a new example and the required
                fields for each example are 'duration' (seconds) is the
                length of the audio, 'key' is the path to the wave file
                and 'text' is the transcription.
            batch_size : Size of the batches for training.
            sample_rate : Rate to resample audio prior to feature computation.
            min_duration : Minimum length of allowed audio in seconds.
            max_duration : Maximum length of allowed audio in seconds.
        """
        self.batch_size = batch_size
        self.sample_rate = sample_rate

        data = []
        data.extend(_read_data_json(data_jsons))

        self.data = data

        # *NB* this cuts off the longest data items in the last segment
        # if len(data) is not a multiple of batch_size
        batches = [data[i:i+batch_size]
                   for i in range(0, len(data) - batch_size + 1, batch_size)]
        random.shuffle(batches)
        self.batches = batches

    def estimate_mean_std(self, sample_size=2048):
        keys =  [random.choice(random.choice(self.batches))['key']
                 for _ in range(sample_size)]
        feats = np.hstack([compute_features(k, self.sample_rate)
                           for k in keys])
        mean = np.mean(feats, axis=1)
        std = np.std(feats, axis=1)
        return mean, std

    def iterator(self, max_size=3,
                 num_workers=3, max_examples=None):
        random.shuffle(self.data)
        self.batches = [self.data[i:i+self.batch_size]
                   for i in range(0, len(self.data) - self.batch_size + 1, self.batch_size)]
        batches = self.batches

        if max_examples is not None:
            batches = batches[:int(max_examples / self.batch_size)]

        consumer = mp.Queue()
        producer = mp.Queue(max_size)
        for b in batches:
            consumer.put(b)

        procs = [mp.Process(target=queue_featurize_example,
                            args=(consumer, producer,
                                  self.sample_rate))
                 for _ in range(num_workers)]
        [p.start() for p in procs]

        for _ in batches:
            yield producer.get()

    @property
    def alphabet_size(self):
        return 1500

    @property
    def input_dim(self):
        return _spec_freq_dim(self.sample_rate)

def queue_featurize_example(consumer, producer, sample_rate):
    while True:
        try:
            batch = consumer.get(block=True, timeout=5)
        except mp.queues.Empty as e:
            return
        labels = []
        inputs = []
        for b in batch:
            labels.append(list(map(int, b['text'].split(','))))
            inputs.append(resample_wav(b['key'], sample_rate))
        producer.put((inputs, labels))

def resample_wav(audio_file, sample_rate):
    y, sr = librosa.load(audio_file, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=128)
    return mfccs

def _read_data_json(file_name):
    with open(file_name, 'r') as fid:
        return [json.loads(l) for l in fid]

def _spec_time_steps(duration, window_size=25, hop_size=10):
    """
    Compute the number of time steps of a spectrogram.

    Args:
        duration : Length of audio in seconds.
        window_size : Size of specgram window in milliseconds.
        hop_size : Size of steps between ffts in
            specgram in milliseconds.
    Returns:
        The number of time-steps in the
        output of the spectrogram.
    """
    duration_ms = duration * 1000
    return math.ceil((duration_ms - window_size) / hop_size)

def _spec_freq_dim(sample_rate, window_size=25):
    """
    Compute the number of frequency bins of a spectrogram.

    Args:
        sample_rate : Hz of the audio.
        window_size : Size of specgram window in milliseconds.
    Returns:
        An integer representing the number of dimensions in
        the spectrogram.
    """
    return int(((sample_rate / 1000) * window_size) / 2) + 1
