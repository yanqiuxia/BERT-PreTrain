# coding=utf-8
__author__ = 'yanqiuxia'

import tensorflow as tf

import modeling
import optimization


class BERTTrainModel(object):

    def __init__(self, model_save_path,bert_config, max_seq_length, max_predictions_per_seq, use_tpu
                 , learning_rate, num_train_steps, num_warmup_steps):

        self.model_save_path = model_save_path
        self.input_ids = tf.placeholder(tf.int64, [None, max_seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int64, [None, max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int64, [None, max_seq_length], name='segment_ids')
        self.masked_lm_positions = tf.placeholder(tf.int64, [None, max_predictions_per_seq], name='masked_lm_positions')
        self.masked_lm_ids = tf.placeholder(tf.int64, [None, max_predictions_per_seq], name='masked_lm_ids')
        self.masked_lm_weights = tf.placeholder(tf.float32, [None, max_predictions_per_seq], name='masked_lm_ids')
        self.next_sentence_labels = tf.placeholder(tf.int64, [None], name='next_sentence_labels')
        # self.is_training = tf.placeholder(tf.bool,[],name='is_training')
        self.hidden_dropout_prob = tf.placeholder(tf.float32, [], name='hidden_dropout_prob')
        self.attention_probs_dropout_prob = tf.placeholder(tf.float32, [], name='attention_probs_dropout_prob')

        model = modeling.BertModel(
            config=bert_config,
            #is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_tpu)

        (self.masked_lm_loss,self.masked_lm_example_loss
         , self.masked_lm_log_probs) = self.get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            self.masked_lm_positions, self.masked_lm_ids, self.masked_lm_weights)

        #去除下一句的预测
        (self.next_sentence_loss, self.next_sentence_example_loss,
         self.next_sentence_log_probs) = self.get_next_sentence_output(
            bert_config, model.get_pooled_output(), self.next_sentence_labels)

        self.total_loss = self.masked_lm_loss + self.next_sentence_loss
        # self.total_loss = self.masked_lm_loss

        self.train_op = optimization.create_optimizer(
            self.total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, var_list=var_list)

    def get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights):
        """Get loss and log probs for the masked LM."""
        input_tensor = self.gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return (loss, per_example_loss, log_probs)

    def get_next_sentence_output(self, bert_config, input_tensor, labels):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            labels = tf.reshape(labels, [-1])
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, per_example_loss, log_probs)

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_offsets = tf.cast(flat_offsets,tf.int64)
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor
