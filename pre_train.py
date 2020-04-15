# coding=utf-8
__author__ = 'yanqiuxia'

import os
import sys
import time
import math
import logging

import numpy as np
import tensorflow as tf
from sklearn import metrics

import modeling
from utils import reader_tfrecord
from bert_train_model import BERTTrainModel

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", './data/chinese_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
## Other parameters
flags.DEFINE_string(
    "init_checkpoint", './data/chinese_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "train_input_file", './data/v0_0_1/train.tf_record',
    "Input TF train example files (can be a glob or comma separated).")

flags.DEFINE_string("dev_input_file", './data/v0_0_1/dev.tf_record',
                    "Input TF dev example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", './output/',
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string('save_path', './output/model.ckpt', 'new model save path')
flags.DEFINE_string('ckpt_path', './output/finetune/', 'finetune checkpoint path')
train_log_file = os.path.join(FLAGS.output_dir, 'train.log')
flags.DEFINE_string('train_log_file', train_log_file, 'train log file')

flags.DEFINE_string('eval_model_path', './output/', 'evaluate model path')
eval_log_file = os.path.join(FLAGS.eval_model_path, 'eval.log')
flags.DEFINE_string('eval_log_file', eval_log_file, 'eval log file')

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_boolean('is_finetune', False, 'whether to finetune the model')

# flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 6, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

# flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_train_epochs", 3,
                     "Total number of training epochs to perform.")

flags.DEFINE_integer("num_warmup_steps", 1000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 5000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class BERTrain(object):

    def __init__(self, train_data_batch, dev_data_batch):
        self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        self.train_data_batch = train_data_batch
        self.dev_data_batch = dev_data_batch
        self.train_batch_num = math.ceil(self.train_data_batch['total_num'] / FLAGS.train_batch_size)
        self.num_train_steps = int(self.train_batch_num * FLAGS.num_train_epochs)

        self.model = BERTTrainModel(
            model_save_path=FLAGS.save_path
            , bert_config=self.bert_config
            , max_seq_length=FLAGS.max_seq_length
            , max_predictions_per_seq=FLAGS.max_predictions_per_seq
            , use_tpu=FLAGS.use_tpu
            , learning_rate=FLAGS.learning_rate
            , num_train_steps=self.num_train_steps
            , num_warmup_steps=FLAGS.num_warmup_steps
        )

    def creat_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def close_session(self):
        # 请求线程结束
        self.coord.request_stop()
        # 等待线程终止
        self.coord.join(self.threads)
        self.sess.close()

    def train(self):

        # 第一种方式
        init_checkpoint = FLAGS.init_checkpoint
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        self.sess.run(tf.global_variables_initializer())

        # #第二种方式
        # tvars = tf.trainable_variables()
        # var_to_restore = [val for val in tvars if 'bert' in val.name]
        # saver = tf.train.Saver(var_to_restore, max_to_keep=3)
        # saver.restore(self.sess, FLAGS.init_checkpoint)

        saver = tf.train.Saver(max_to_keep=3)
        if FLAGS.is_finetune:
            checkpoint = tf.train.latest_checkpoint(FLAGS.ckpt_path)
            saver.restore(self.sess, checkpoint)

        display_num = 1  # Display 5 pre epoch
        train_batch_num = self.train_batch_num
        # display_batch = int(train_batch_num / display_num)

        num_train_epochs = FLAGS.num_train_epochs
        train_steps = 0
        best_F1 = 0.0
        for epoch in range(num_train_epochs):
            start_time = time.time()
            total_loss = 0
            for batch in range(train_batch_num):
                train_steps += 1
                fetches = [self.model.total_loss, self.model.train_op]

                input_ids = self.train_data_batch['input_ids']
                input_mask = self.train_data_batch['input_mask']
                segment_ids = self.train_data_batch['segment_ids']
                masked_lm_positions = self.train_data_batch['masked_lm_positions']
                masked_lm_ids = self.train_data_batch['masked_lm_ids']
                masked_lm_weights = self.train_data_batch['masked_lm_weights']
                next_sentence_labels = self.train_data_batch['next_sentence_labels']

                input_ids, input_mask, segment_ids, masked_lm_positions \
                    , masked_lm_ids, masked_lm_weights, next_sentence_labels = self.sess.run([
                    input_ids, input_mask, segment_ids, masked_lm_positions
                    , masked_lm_ids, masked_lm_weights, next_sentence_labels
                ])

                hidden_dropout_prob = self.bert_config.hidden_dropout_prob
                attention_probs_dropout_prob = self.bert_config.attention_probs_dropout_prob
                feed_dict = {
                    self.model.input_ids: input_ids,
                    self.model.input_mask: input_mask,
                    self.model.segment_ids: segment_ids,
                    self.model.masked_lm_positions: masked_lm_positions,
                    self.model.masked_lm_ids: masked_lm_ids,
                    self.model.masked_lm_weights: masked_lm_weights,
                    self.model.next_sentence_labels: next_sentence_labels,
                    self.model.hidden_dropout_prob: hidden_dropout_prob,
                    self.model.attention_probs_dropout_prob: attention_probs_dropout_prob,
                }

                [loss, _] = self.sess.run(fetches, feed_dict)
                total_loss += loss

                if batch % 1000 == 0:
                    print("== epoch: %d/%d, batch: %d/%d, training loss: %g " % (
                    epoch + 1, num_train_epochs, batch, train_batch_num, loss))

                if (batch + 1) % FLAGS.iterations_per_loop == 0:
                    '''
                    做评估
                    '''
                    eval_metrics = self._evaluate(self.dev_data_batch)
                    for key in sorted(eval_metrics.keys()):
                        tf.logging.info("  %s = %s", key, str(eval_metrics[key]))
                    avg_F1 = eval_metrics['masked_lm_F1'] + eval_metrics['next_sentence_F1']
                    if avg_F1 > best_F1:
                        save_path = saver.save(self.sess, self.model.model_save_path, global_step=train_steps)
                        print('the save path is ', save_path)
                        best_F1 = avg_F1

                # if (train_steps) % FLAGS.save_checkpoints_steps == 0:  # Save once per save_checkpoints_steps
                #     save_path = saver.save(self.sess, self.model.model_save_path, global_step=train_steps)
                #     print('the save path is ', save_path)

            mean_loss = total_loss / train_batch_num

            # 最后一步模型的数据保存
            if train_steps == self.num_train_steps:
                save_path = saver.save(self.sess, self.model.model_save_path, global_step=train_steps)
                print('the save path is ', save_path)

            end_time = time.time()
            run_time = end_time - start_time
            print('Epoch training num %d, loss=%g, speed=%g s/epoch' % (
                self.train_data_batch['total_num'], mean_loss, run_time))

        # 最优结果模型的评估
        eval_metrics = self._evaluate(self.dev_data_batch)
        for key in sorted(eval_metrics.keys()):
            tf.logging.info("  %s = %s", key, str(eval_metrics[key]))

    def _evaluate(self, dataset=None):

        # 取下整
        batch_num = int(dataset['total_num'] / FLAGS.eval_batch_size)
        masked_lm_mean_loss = 0
        masked_lm_accuracy = 0
        masked_lm_P = 0
        masked_lm_R = 0
        masked_lm_F1 = 0

        next_sentence_mean_loss = 0
        next_sentence_accuracy = 0
        next_sentence_P = 0
        next_sentence_R = 0
        next_sentence_F1 = 0

        for batch in range(batch_num):
            input_ids = dataset['input_ids']
            input_mask = dataset['input_mask']
            segment_ids = dataset['segment_ids']
            masked_lm_positions = dataset['masked_lm_positions']
            masked_lm_ids = dataset['masked_lm_ids']
            masked_lm_weights = dataset['masked_lm_weights']
            next_sentence_labels = dataset['next_sentence_labels']

            input_ids, input_mask, segment_ids, masked_lm_positions \
                , masked_lm_ids, masked_lm_weights, next_sentence_labels = self.sess.run([
                input_ids, input_mask, segment_ids, masked_lm_positions
                , masked_lm_ids, masked_lm_weights, next_sentence_labels
            ])

            feed_dict = {
                self.model.input_ids: input_ids,
                self.model.input_mask: input_mask,
                self.model.segment_ids: segment_ids,
                self.model.masked_lm_positions: masked_lm_positions,
                self.model.masked_lm_ids: masked_lm_ids,
                self.model.masked_lm_weights: masked_lm_weights,
                self.model.next_sentence_labels: next_sentence_labels,
                self.model.hidden_dropout_prob: 0.0,
                self.model.attention_probs_dropout_prob: 0.0,
            }
            fetches = [self.model.masked_lm_log_probs, self.model.masked_lm_example_loss
                , self.model.next_sentence_log_probs, self.model.next_sentence_example_loss
                , self.model.total_loss]
            [masked_lm_log_probs, masked_lm_example_loss
                , next_sentence_log_probs, next_sentence_example_loss
                , loss] = self.sess.run(fetches, feed_dict)

            if batch % 1000 == 0:
                print("batch: %d/%d, eval loss: %g " % (batch, batch_num, loss))

            masked_lm_log_probs = np.reshape(masked_lm_log_probs,
                                             [-1, masked_lm_log_probs.shape[-1]])
            masked_lm_predictions = np.argmax(
                masked_lm_log_probs, axis=-1)
            masked_lm_example_loss = np.reshape(masked_lm_example_loss, [-1])
            masked_lm_ids = np.reshape(masked_lm_ids, [-1])
            masked_lm_weights = np.reshape(masked_lm_weights, [-1])
            masked_lm_accuracy += metrics.accuracy_score(masked_lm_ids, masked_lm_predictions
                                                         , normalize=True, sample_weight=masked_lm_weights)
            masked_lm_P += metrics.precision_score(masked_lm_ids, masked_lm_predictions
                                                   , sample_weight=masked_lm_weights, average='micro')
            masked_lm_R += metrics.recall_score(masked_lm_ids, masked_lm_predictions
                                                , sample_weight=masked_lm_weights, average='micro')
            masked_lm_F1 += metrics.f1_score(masked_lm_ids, masked_lm_predictions
                                             , sample_weight=masked_lm_weights, average='micro')

            next_sentence_log_probs = np.reshape(
                next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
            next_sentence_predictions = np.argmax(
                next_sentence_log_probs, axis=-1)

            next_sentence_labels = np.reshape(next_sentence_labels, [-1])

            next_sentence_accuracy += metrics.accuracy_score(next_sentence_labels, next_sentence_predictions
                                                             , normalize=True)
            next_sentence_P += metrics.precision_score(next_sentence_labels, next_sentence_predictions)
            next_sentence_R += metrics.recall_score(next_sentence_labels, next_sentence_predictions)
            next_sentence_F1 += metrics.f1_score(next_sentence_labels, next_sentence_predictions)

            weight_masked_lm_loss = masked_lm_example_loss * masked_lm_weights
            temp_masked_lm_mean_loss = np.mean(weight_masked_lm_loss)
            masked_lm_mean_loss += temp_masked_lm_mean_loss

            temp_next_sentence_mean_loss = np.mean(next_sentence_example_loss)
            next_sentence_mean_loss += temp_next_sentence_mean_loss

        masked_lm_accuracy = masked_lm_accuracy / batch_num
        masked_lm_mean_loss = masked_lm_mean_loss / batch_num
        next_sentence_accuracy = next_sentence_accuracy / batch_num
        next_sentence_mean_loss = next_sentence_mean_loss / batch_num

        masked_lm_P = masked_lm_P / batch_num
        masked_lm_R = masked_lm_R / batch_num
        # masked_lm_F1 = 2*masked_lm_P*masked_lm_R / (masked_lm_P+masked_lm_R)
        masked_lm_F1 = masked_lm_F1 / batch_num

        next_sentence_P = next_sentence_P / batch_num
        next_sentence_R = next_sentence_R / batch_num
        # next_sentence_F1 = 2*next_sentence_P*next_sentence_R / (next_sentence_P+next_sentence_R)

        next_sentence_F1 = next_sentence_F1 / batch_num

        avg_F1 = (masked_lm_F1 + next_sentence_F1) / 2
        eval_metrics = {
            "eval_num": dataset['total_num'],
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_P": masked_lm_P,
            "masked_lm_R": masked_lm_R,
            "masked_lm_F1": masked_lm_F1,
            "masked_lm_mean_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_mean_loss": next_sentence_mean_loss,
            "next_sentence_P": next_sentence_P,
            "next_sentence_R": next_sentence_R,
            "next_sentence_F1": next_sentence_F1,
            "avg_F1": avg_F1,
        }

        return eval_metrics

    def evaluate(self, dataset=None):

        # # 第一种方式
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        checkpoint = tf.train.latest_checkpoint(FLAGS.eval_model_path)
        # checkpoint = FLAGS.init_checkpoint
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint)
        tf.train.init_from_checkpoint(checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        self.sess.run(tf.global_variables_initializer())

        # 第二种方式  只能加载bert参数
        # tvars = tf.trainable_variables()
        # var_to_restore = [val for val in tvars if 'bert' in val.name]
        # saver = tf.train.Saver(var_to_restore)
        # saver = tf.train.Saver()
        #
        # checkpoint = tf.train.latest_checkpoint(FLAGS.eval_model_path)
        # checkpoint = './chinese_L-12_H-768_A-12/bert_model.ckpt'
        # checkpoint = './output/eval_ckpt/model.ckpt-15969'
        # saver.restore(self.sess, checkpoint)

        tf.logging.info(checkpoint)

        # 调用评估函数
        result = self._evaluate(dataset)
        output_eval_file = os.path.join(FLAGS.eval_model_path, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


def main(_):
    '''

    :param _:
    :return:
    '''

    tf.gfile.MakeDirs(FLAGS.output_dir)
    if FLAGS.do_train:
        handlers = [
            logging.FileHandler(FLAGS.train_log_file),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        handlers = [
            logging.FileHandler(FLAGS.eval_log_file),
            logging.StreamHandler(sys.stdout)
        ]

    logging.getLogger('tensorflow').handlers = handlers
    tf.logging.set_verbosity(tf.logging.INFO)

    train_data_batch = reader_tfrecord(FLAGS.train_input_file, FLAGS.max_seq_length
                                       , FLAGS.max_predictions_per_seq, FLAGS.train_batch_size
                                       , capacity=64, min_after_dequeue=10, mode='train')

    dev_data_batch = reader_tfrecord(FLAGS.dev_input_file, FLAGS.max_seq_length
                                     , FLAGS.max_predictions_per_seq, FLAGS.eval_batch_size
                                     , capacity=64, min_after_dequeue=10, mode='eval')

    tf.logging.info('Read tf records data completion')
    tf.logging.info('train data num:%d, dev data num:%d'%(train_data_batch['total_num'],dev_data_batch['total_num']))

    bert_train = BERTrain(train_data_batch=train_data_batch, dev_data_batch=dev_data_batch)

    bert_train.creat_session()

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        bert_train.train()

    else:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        bert_train.evaluate(dev_data_batch)

    bert_train.close_session()


if __name__ == "__main__":
    tf.app.run()
