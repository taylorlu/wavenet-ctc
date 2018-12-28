"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, optimizer_factory
import sys
sys.path.append('wavenet')
sys.path.append('user_ops')
import audio_producer
import numpy as np

BATCH_SIZE = 32
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False
MAX_GRAD_NORM = 50

DECAY_STEPS = int(102600/BATCH_SIZE)
DECAY_RATE = 0.9


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def _batch_major(data):
    """
    Reshapes a batch of spectrogram arrays into
    a single tensor [batch_size x max_time x input_dim].

    Args :
        data : list of 2D numpy arrays of [input_dim x time]
    Returns :
        A 3d tensor with shape
        [batch_size x max_time x input_dim]
        and zero pads as necessary for data items which have
        fewer time steps than max_time.
    """
    max_time = max(d.shape[1] for d in data)
    batch_size = len(data)
    input_dim = data[0].shape[0]
    all_data = np.zeros((batch_size, max_time, input_dim),
                        dtype=np.float32)
    for e, d in enumerate(data):
        all_data[e, :d.shape[1], :] = d.T
    return all_data


def main():
    args = get_arguments()

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Create network.
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        word_count=wavenet_params["word_count"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"],
        histograms=args.histograms,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=None)

    _audio_batch = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, None, 40])
    _sequence_lengths = tf.placeholder(tf.int32, shape=args.batch_size)
    _labels = tf.placeholder(tf.int32)
    _label_lens = tf.placeholder(tf.int32)

    gc_id_batch = None


    loss = net.ctc_loss(input_batch=_audio_batch,sequence_lengths=_sequence_lengths,
                    labels=_labels,label_lens=_label_lens,
                    global_condition_batch=gc_id_batch,
                    l2_regularization_strength=args.l2_regularization_strength)
    input_batch = net.get_input_batch()


    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(args.learning_rate, global_step,
                DECAY_STEPS, DECAY_RATE, staircase=True)

    ema = tf.train.ExponentialMovingAverage(0.99, name="avg")
    avg_cost_op = ema.apply([loss])

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    scaled_grads, norm = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)

    optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-4)
    with tf.control_dependencies([avg_cost_op]):
        train_op = optimizer.apply_gradients(zip(scaled_grads, tvars),
                             global_step=global_step)


    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    train_jsons = 'trainSamples.json'
    saveDir = './save2/model'
    producer = audio_producer.AudioProducer(train_jsons, args.batch_size, sample_rate=wavenet_params['sample_rate'])

    try:
        saver = tf.train.Saver(max_to_keep=1)
        # saver.restore(sess, saveDir)

        for epoch in range(100):
            for e, (inputs, labels) in enumerate(producer.iterator()):

                sequence_lengths = [d.shape[1] for d in inputs]
                inputs = _batch_major(inputs)
                pad_size = net.receptive_field-1
                head_pad = int(pad_size/2)
                tail_pad = pad_size - head_pad
                inputs = np.pad(inputs, [[0, 0], [head_pad, tail_pad], [0, 0]], 'constant')

                values = [l for label in labels for l in label]
                label_lens = [len(label) for label in labels]

                feed_dict = { _audio_batch : inputs,
                                _sequence_lengths : sequence_lengths,
                                _labels : values,
                                _label_lens : label_lens}

                loss_value, _ , _input_batch, _global_step, _lr = sess.run(
                        [loss, train_op, input_batch, global_step, lr], feed_dict=feed_dict)
                print("epoch = {}, global_step = {}, lr = {}, loss = {}".format(epoch, _global_step, _lr, loss_value))
                if(_global_step%1000==0 and _global_step!=0):
                    saver.save(sess, saveDir)



    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()


if __name__ == '__main__':
    main()
