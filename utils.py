# coding=utf-8
__author__ = 'yanqiuxia'

import tensorflow as tf

# 该函数用于统计 TFRecord 文件中的样本数量(总数)
def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return  sample_nums


def reader_tfrecord(file_name, max_seq_length, max_predictions_per_seq,batch_size,capacity,min_after_dequeue,mode):

    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.FixedLenFeature([], tf.int64),
    }
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=name_to_features)
    input_ids = tf.cast(features['input_ids'], tf.int64)
    input_mask = tf.cast(features['input_mask'], tf.int64)
    segment_ids = tf.cast(features['segment_ids'], tf.int64)
    masked_lm_positions = tf.cast(features['masked_lm_positions'], tf.int64)
    masked_lm_ids = tf.cast(features['masked_lm_ids'], tf.int64)
    masked_lm_weights = tf.cast(features['masked_lm_weights'], tf.float32)
    next_sentence_labels = tf.cast(features['next_sentence_labels'], tf.int64)

    if mode=='train':

        input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch\
        , masked_lm_ids_batch, masked_lm_weights_batch, next_sentence_labels_batch = tf.train.shuffle_batch([input_ids, input_mask,segment_ids
                                                ,masked_lm_positions,masked_lm_ids,masked_lm_weights
                                             ,next_sentence_labels], batch_size=batch_size
                                            , capacity=capacity, min_after_dequeue=min_after_dequeue
                                            ,allow_smaller_final_batch=True)
    if mode=='eval':
        input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch \
            , masked_lm_ids_batch, masked_lm_weights_batch, next_sentence_labels_batch = tf.train.batch(
            [input_ids, input_mask, segment_ids
                , masked_lm_positions, masked_lm_ids, masked_lm_weights
                , next_sentence_labels], batch_size=batch_size
               , capacity=capacity, allow_smaller_final_batch=False)

    total_num = total_sample(file_name)
    # batch_num = math.ceil(total_num/batch_size)

    data_batch = {}
    data_batch['input_ids'] = input_ids_batch
    data_batch['input_mask'] = input_mask_batch
    data_batch['segment_ids'] = segment_ids_batch
    data_batch['masked_lm_positions'] = masked_lm_positions_batch
    data_batch['masked_lm_ids'] = masked_lm_ids_batch
    data_batch['masked_lm_weights'] = masked_lm_weights_batch
    data_batch['next_sentence_labels'] = next_sentence_labels_batch
    data_batch['total_num'] = total_num

    return data_batch







