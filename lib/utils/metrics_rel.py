# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import datetime
import logging
import os
from collections import OrderedDict

from core.config_rel import cfg
from utils import checkpoints_rel
from caffe2.python import workspace
from utils import helpers_rel

logger = logging.getLogger(__name__)


class MetricsCalculator():

    def __init__(self, model, split, batch_size, prefix):
        self.model = model
        self.split = split
        self.prefix = prefix
        self.batch_size = batch_size
        # for any dataset, we have two choices:
        #   1. Compute metrics over all the classes
        #   2. Compute metrics over first few classes only
        # We can compute all metrics or chose to compute only one. The chosen
        # metrics will be computed for train/test/val dataset.
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            self.best_top1 = float('inf')
            self.best_top5 = float('inf')
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            self.best_top1_N_way = float('inf')
            self.best_top5_N_way = float('inf')
        self.reset()

    def reset(self):
        # this should clear out all the metrics computed so far except the
        # best_topN metrics
        logger.info('Resetting {} metrics...'.format(self.split))
        self.split_loss = 0.0
        self.split_N = 0
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            self.split_err = 0.0
            self.split_err5 = 0.0
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            self.split_err_N_way = 0.0
            self.split_err5_N_way = 0.0

    def finalize_metrics(self):
        self.split_loss /= self.split_N
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            self.split_err /= self.split_N
            self.split_err5 /= self.split_N
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            self.split_err_N_way /= self.split_N
            self.split_err5_N_way /= self.split_N

    def get_computed_metrics(self):
        json_stats = {}
        if self.split == 'train':
            json_stats['train_loss'] = round(self.split_loss, 3)
            if cfg.METRICS.EVALUATE_ALL_CLASSES:
                json_stats['train_accuracy'] = round(100 - self.split_err, 3)
                json_stats['train_err'] = round(self.split_err, 3)
                json_stats['train_err5'] = round(self.split_err5, 3)
            if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
                json_stats['train_err_N_way'] = round(self.split_err_N_way, 3)
                json_stats['train_err5_N_way'] = round(self.split_err5_N_way, 3)
                json_stats['train_accuracy_N_way'] = round(
                    100 - self.split_err_N_way, 3)
        elif self.split in ['test', 'val']:
            if cfg.METRICS.EVALUATE_ALL_CLASSES:
                json_stats['test_accuracy'] = round(100 - self.split_err, 3)
                json_stats['test_err'] = round(self.split_err, 3)
                json_stats['test_err5'] = round(self.split_err5, 3)
                json_stats['best_accuracy'] = round(100 - self.best_top1, 3)
                json_stats['best_err'] = round(self.best_top1, 3)
                json_stats['best_err5'] = round(self.best_top5, 3)
            if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
                json_stats['test_accuracy_N_way'] = round(
                    100 - self.split_err_N_way, 3)
                json_stats['test_err_N_way'] = round(self.split_err_N_way, 3)
                json_stats['test_err5_N_way'] = round(self.split_err5_N_way, 3)
                json_stats['best_err_N_way'] = round(self.best_top1_N_way, 3)
                json_stats['best_err5_N_way'] = round(self.best_top5_N_way, 3)
        return json_stats

    # this is valid only for the test data
    def log_best_model_metrics(
        self, model_iter, total_iters, median_best_err, median_best_err_N_way,
    ):
        best_model_metrics_str = self.get_best_model_metrics_str()
        if cfg.METRICS.EVALUATE_ALL_CLASSES and cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            print(best_model_metrics_str.format(
                model_iter + 1, total_iters, self.best_top1, self.best_top5,
                self.best_top1_N_way, self.best_top5_N_way
            ))
        elif cfg.METRICS.EVALUATE_ALL_CLASSES:
            print(best_model_metrics_str.format(
                model_iter + 1, total_iters, self.best_top1, self.best_top5
            ))
        elif cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            print(best_model_metrics_str.format(
                model_iter + 1, total_iters, self.best_top1_N_way,
                self.best_top5_N_way
            ))
        if cfg.METRICS.NUM_MEDIAN_EPOCHS > 0:
            if cfg.METRICS.EVALUATE_ALL_CLASSES:
                median_err = np.median(np.array(median_best_err))
                print('\n* Finished median best_err for {} last epochs: {:7.3f}\n'
                        .format(cfg.METRICS.NUM_MEDIAN_EPOCHS, median_err))
            if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
                median_err_N_way = np.median(np.array(median_best_err_N_way))
                print('\n* Finished median best_err_N_way for {} last epochs: {:7.3f}\n'
                        .format(cfg.METRICS.NUM_MEDIAN_EPOCHS, median_err_N_way))

    # This prints the metrics of an epoch
    def log_final_metrics(self, model_iter, total_iters):
        final_metrics = self.get_final_metrics_log_str()
        if cfg.METRICS.EVALUATE_ALL_CLASSES and cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            print(final_metrics.format(
                model_iter + 1, total_iters, self.split_err, self.split_err5,
                self.split_err_N_way, self.split_err5_N_way
            ))
        elif cfg.METRICS.EVALUATE_ALL_CLASSES:
            print(final_metrics.format(
                model_iter + 1, total_iters, self.split_err, self.split_err5
            ))
        elif cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            print(final_metrics.format(
                model_iter + 1, total_iters, self.split_err_N_way,
                self.split_err5_N_way
            ))

    def compute_log_and_checkpoint_best_topN(
        self, model, model_iter, checkpoint_dir=None
    ):
        if cfg.METRICS.EVALUATE_ALL_CLASSES and self.split_err < self.best_top1:
            self.best_top1 = self.split_err
            self.best_top5 = self.split_err5
            print('\n* Best model: top1: {:7.3f} top5: {:7.3f}\n'.format(
                self.best_top1, self.best_top5
            ))
            if checkpoint_dir is not None:
                params_file = os.path.join(checkpoint_dir, 'bestModel.pkl')
                checkpoints_rel.save_model_params(
                    model=model, params_file=params_file, model_iter=model_iter,
                    checkpoint_dir=checkpoint_dir
                )
        if (cfg.METRICS.EVALUATE_FIRST_N_WAYS and
                self.split_err_N_way < self.best_top1_N_way):
            self.best_top1_N_way = self.split_err_N_way
            self.best_top5_N_way = self.split_err5_N_way
            print('\n* Best model: top1_N_way: {:7.3f} top5_N_way: {:7.3f}\n'
                    .format(self.best_top1_N_way, self.best_top5_N_way))
            if cfg.CHECKPOINT.CHECKPOINT_MODEL:
                params_file = os.path.join(
                    checkpoint_dir, 'bestModel_N_way.pkl')
                checkpoints_rel.save_model_params(
                    model=model, params_file=params_file,
                    model_iter=model_iter, checkpoint_dir=checkpoint_dir
                )
        # best_topN is calculated only when training happens
        self.log_final_metrics(model_iter, cfg.SOLVER.NUM_ITERATIONS)

    def get_split_err(self, input_db):
        split_err, split_err5 = None, None
        split_err_N_way, split_err5_N_way = None, None
        accuracy_metrics = compute_multi_device_topk_accuracy(
            top_k=1, split=self.split, input_db=input_db
        )
        accuracy5_metrics = compute_multi_device_topk_accuracy(
            top_k=5, split=self.split, input_db=input_db
        )
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            split_err = (1.0 - accuracy_metrics['topk_accuracy']) * 100
            split_err5 = (1.0 - accuracy5_metrics['topk_accuracy']) * 100
            self.split_err += (split_err * self.batch_size)
            self.split_err5 += (split_err5 * self.batch_size)
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            split_err_N_way = (1.0 - accuracy_metrics['topk_N_way_accuracy']) * 100
            split_err5_N_way = (1.0 - accuracy5_metrics['topk_N_way_accuracy']) * 100
            self.split_err_N_way += (split_err_N_way * self.batch_size)
            self.split_err5_N_way += (split_err5_N_way * self.batch_size)
        return split_err, split_err5, split_err_N_way, split_err5_N_way

    def sum_multi_gpu_blob(self, blob_name):
        """Return the sum of a scalar blob held on multiple GPUs."""
        prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
        val = 0
        for gpu_id in range(cfg.ROOT_DEVICE_ID, cfg.ROOT_DEVICE_ID + cfg.NUM_DEVICES):
            val += \
                float(workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, blob_name)))
        return val

    def average_multi_gpu_blob(self, blob_name):
        """Return the average of a scalar blob held on multiple GPUs."""
        return self.sum_multi_gpu_blob(blob_name) / cfg.NUM_DEVICES

    def calculate_and_log_train_metrics(
        self, losses, metrics, curr_iter, timer, rem_iters, total_iters, mb_size=None,
        db_loader=None, input_db=None
    ):
        self.lr = round(float(workspace.FetchBlob(self.prefix + '/lr')), 16)
        iter_losses = OrderedDict()
        for k in losses:
            key = str(k)
            if key.find('/') >= 0:
                key = key.split('/')[1]
            iter_losses[key] = self.sum_multi_gpu_blob(key)
            # iter_losses[key] = self.average_multi_gpu_blob(key)
        # iter_losses['total_loss'] = np.sum(np.array(iter_losses.values()))
        split_loss = np.sum(np.array(iter_losses.values())) * \
            cfg.MODEL.GRAD_ACCUM_FREQUENCY
        self.split_loss += (split_loss * self.batch_size)
        iter_metrics = OrderedDict()
        for k in metrics:
            key = str(k)
            if key.find('/') >= 0:
                key = key.split('/')[1]
            # logger.info('metric name: {}'.format(key))
            iter_metrics[key] = self.average_multi_gpu_blob(key)
        if (curr_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_str = self.get_iter_metrics_log_str(self.split)

            for k in iter_losses.keys():
                log_str += (', ' + k + ': {:7.4f}')
            for k in iter_metrics.keys():
                log_str += (', ' + k + ': {:7.4f}')
            iter_values = iter_losses.values() + iter_metrics.values()
            print(log_str.format(eta, self.lr, curr_iter + 1, total_iters,
                  timer.diff, timer.average_time, split_loss, mb_size,
                  *iter_values))
        self.split_N += self.batch_size

    def calculate_and_log_test_metrics(
        self, curr_iter, timer, rem_iters, total_iters, mb_size=None,
        db_loader=None, input_db=None
    ):
        self.lr = cfg.SOLVER.BASE_LR
        self.split_N += self.batch_size
        split_err, split_err5, split_err_N_way, split_err5_N_way = self.get_split_err(
            input_db)
        if (curr_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_str = self.get_iter_metrics_log_str(self.split)
            if (cfg.METRICS.EVALUATE_ALL_CLASSES and
                    cfg.METRICS.EVALUATE_FIRST_N_WAYS):
                print(log_str.format(
                    eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                    timer.average_time, split_err, self.split_err / self.split_N,
                    split_err5, self.split_err5 / self.split_N, split_err_N_way,
                    self.split_err_N_way / self.split_N, split_err5_N_way,
                    self.split_err5_N_way / self.split_N
                ))
            elif cfg.METRICS.EVALUATE_ALL_CLASSES:
                print(log_str.format(
                    eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                    timer.average_time, split_err, self.split_err / self.split_N,
                    split_err5, self.split_err5 / self.split_N
                ))
            elif cfg.METRICS.EVALUATE_FIRST_N_WAYS:
                print(log_str.format(
                    eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                    timer.average_time, split_err_N_way,
                    self.split_err_N_way / self.split_N, split_err5_N_way,
                    self.split_err5_N_way / self.split_N
                ))

    def get_best_model_metrics_str(self):
        best_model_metrics_str = '\n* Best val model [{}|{}]:\t'
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            best_model_metrics_str += ' top1: {:7.3f} top5: {:7.3f}'
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            best_model_metrics_str += '  top1_N_way {:7.3f} top5_N_way {:7.3f}\n'
        return best_model_metrics_str

    def get_final_metrics_log_str(self):
        final_output_str = '\n* Finished #iters [{}|{}]:\t'
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            final_output_str += ' top1: {:7.3f} top5: {:7.3f}'
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            final_output_str += '  top1_N_way {:7.3f} top5_N_way {:7.3f}\n'
        return final_output_str

    def get_iter_metrics_log_str(self, split):
        train_str = ' '.join((
            '| Train ETA: {}, LR: {}, Iters: [{}/{}], Time: {:0.3f}, AvgTime: {:0.3f},',
            ' Loss: {:7.4f}'
        ))
        test_str = '| Test ETA: {} LR: {} [{}/{}] Time {:0.3f} AvgTime {:0.3f}'
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            train_str += ' top1 {:7.3f} top5 {:7.3f}'
            test_str += '  top1 {:7.3f} ({:7.3f}) top5 {:7.3f} ({:7.4f})'
        # For reporting any new metrics, amend the str accordingly
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            train_str += '  top1_N_way {:7.3f} top5_N_way {:7.3f}'
            test_str += '  top1_N_way {:7.3f} ({:7.3f})  top5_N_way {:7.3f} ({:7.4f})'
        if split == 'train':
            train_str += ', mb_size: {}'
            return train_str
        else:
            return test_str

    def get_layer_stats(self, model):
        # we will compute the layer stats for layers described in STATS_LAYERS
        # the stats computed are: p15, p50, p85, mean
        # the stats should be computed for the activations, grads, weights of
        # those layers
        layer_stats = {}
        ws_blobs = workspace.Blobs()
        num_blobs = int(len(ws_blobs) / cfg.NUM_DEVICES)
        master_device_blobs = ws_blobs[:num_blobs]
        for blob in master_device_blobs:
            for layer in cfg.STATS_LAYERS:
                if blob.find(layer) >= 0:
                    blob_value = (
                        workspace.FetchBlob(blob).reshape((-1)).astype(np.float)
                    )
                    bl_name = helpers_rel.unscope_name(blob)
                    bl_mean = np.mean(blob_value)
                    bl_p15 = np.percentile(blob_value, 15)
                    bl_p50 = np.percentile(blob_value, 50)
                    bl_p85 = np.percentile(blob_value, 85)
                    layer_stats[bl_name + '_p15'] = bl_p15
                    layer_stats[bl_name + '_p50'] = bl_p50
                    layer_stats[bl_name + '_p85'] = bl_p85
                    layer_stats[bl_name + '_mean'] = bl_mean
                    break
        return layer_stats


def update_median_err(
    json_stats, curr_iter, median_best_err, median_best_err_N_way
):
    total_epochs = cfg.SOLVER.NUM_ITERATIONS / cfg.TRAIN.EVALUATION_FREQUENCY
    curr_epoch = json_stats['epoch']
    if curr_epoch > (total_epochs - cfg.METRICS.NUM_MEDIAN_EPOCHS):
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            median_best_err_N_way.append(json_stats['test_err_N_way'])
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            median_best_err.append(json_stats['test_err'])
    return median_best_err, median_best_err_N_way


def get_json_stats_dict(
    train_metrics_calculator, test_metrics_calculator, curr_iter, model_flops,
    model_params,
):
    used_gpu_memory = None
    if cfg.DEVICE == 'GPU':
        used_gpu_memory = helpers_rel.get_gpu_stats()
    json_stats = dict(
        model_flops=model_flops,
        model_params=model_params,
        currentIter=(curr_iter + 1),
        epoch=(1 + int(curr_iter / cfg.TRAIN.EVALUATION_FREQUENCY)),
        used_gpu_memory=used_gpu_memory,
        warmup_type=cfg.SOLVER.WARM_UP_TYPE,
        bn_init=cfg.MODEL.BN_INIT_GAMMA,
        evaluation_frequency=cfg.TRAIN.EVALUATION_FREQUENCY,
        batchSize=cfg.TRAIN.IMS_PER_BATCH,
        nEpochs=int(cfg.SOLVER.NUM_ITERATIONS / cfg.TRAIN.EVALUATION_FREQUENCY),
        dataset=cfg.DATASET,
        model_name=cfg.MODEL.MODEL_NAME,
        num_classes=cfg.MODEL.NUM_CLASSES,
        depth=cfg.MODEL.DEPTH,
        momentum=cfg.SOLVER.MOMENTUM,
        weightDecay=cfg.SOLVER.WEIGHT_DECAY,
        nGPU=cfg.NUM_DEVICES,
        LR=cfg.SOLVER.BASE_LR,
        tenCrop=cfg.TEST.TEN_CROP,
        bn_momentum=cfg.MODEL.BN_MOMENTUM,
        bn_epsilon=cfg.MODEL.BN_EPSILON,
        current_learning_rate=train_metrics_calculator.lr,
    )
    computed_train_metrics = train_metrics_calculator.get_computed_metrics()
    json_stats.update(computed_train_metrics)
    if cfg.DISTRIBUTED.DISTR_ON:
        json_stats['num_nodes'] = cfg.DISTRIBUTED.NUM_NODES
        json_stats['distr_engine'] = cfg.DISTRIBUTED.ALL_REDUCE_ENGINE
        json_stats['random_sampling'] = cfg.DISTRIBUTED.RANDOM_SAMPLE
        json_stats['global_shuffle'] = cfg.DISTRIBUTED.GLOBAL_SHUFFLE
    if test_metrics_calculator is not None:
        computed_test_metrics = test_metrics_calculator.get_computed_metrics()
        json_stats.update(computed_test_metrics)
    return json_stats


def compute_topk_accuracy(top_k, preds, labels, input_db=None, paths=None):
    batch_size = preds.shape[0]
    for i in range(batch_size):
        preds[i, :] = np.argsort(-preds[i, :])
    top_k_preds = preds[:, :top_k]

    pred_labels = []
    for j in range(batch_size):
        pred_labels.append(int(
            labels[j] in top_k_preds[j, :].astype(np.int32).tolist()
        ))
    correct = sum(pred_labels)
    return (float(correct) / batch_size)


def compute_multi_device_topk_accuracy(top_k, split, input_db=None):
    top_k_accuracy = 0.0
    topk_N_way_accuracy = 0.0
    root_device_id = cfg.ROOT_DEVICE_ID
    num_devices = cfg.NUM_DEVICES
    ten_crop = cfg.TEST.TEN_CROP
    computed_metrics = {}
    device_prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    for idx in range(root_device_id, root_device_id + num_devices):
        prefix = '{}{}'.format(device_prefix, idx)
        softmax = workspace.FetchBlob(prefix + '/pred')
        if ten_crop and split in ['test', 'val']:
            softmax = np.reshape(
                softmax, (softmax.shape[0] / 10, 10, softmax.shape[1]))
            softmax = np.mean(softmax, axis=1)
        labels = workspace.FetchBlob(prefix + '/labels')
        paths = None
        batch_size = softmax.shape[0]
        assert labels.shape[0] == batch_size, \
            "Something went wrong with data loading"
        if cfg.METRICS.EVALUATE_ALL_CLASSES:
            top_k_accuracy += compute_topk_accuracy(
                top_k, softmax.copy(), labels, input_db, paths
            )
        if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
            softmax = softmax[:, 0:cfg.METRICS.FIRST_N_WAYS]
            topk_N_way_accuracy += compute_topk_accuracy(
                top_k, softmax.copy(), labels, input_db, paths
            )
    if cfg.METRICS.EVALUATE_ALL_CLASSES:
        computed_metrics['topk_accuracy'] = float(top_k_accuracy) / num_devices
    if cfg.METRICS.EVALUATE_FIRST_N_WAYS:
        computed_metrics['topk_N_way_accuracy'] = (
            float(topk_N_way_accuracy) / num_devices
        )
    return computed_metrics


def sum_multi_device_blob(blob_name, num_devices):
    """Average values of a blob on each device"""
    value = 0
    root_device_id = cfg.ROOT_DEVICE_ID
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    for idx in range(root_device_id, root_device_id + num_devices):
        value += workspace.FetchBlob('{}{}/{}'.format(prefix, idx, blob_name))
    return value


def average_multi_device_blob(blob_name, num_devices):
    """Average values of a blob on each device"""
    value = sum_multi_device_blob(blob_name, num_devices)
    return value / num_devices
