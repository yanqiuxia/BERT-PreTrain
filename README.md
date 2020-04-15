# BERT-PreTrain
不采用tensorflow estimator，分别采用字mask和wwm mask在中文领域内finetune bert模型。

 [论文地址](https://arxiv.org/pdf/1810.04805.pdf)

 [google bert源码](https://github.com/google-research/bert)


# BERT 预训练

bert 除了官方论文

如果需要在各个版本的bert-base、bert-wwm、 bert-wwm-ext、 Roberta-wwm-ext 模型中，采用新领域中的数据进行预训练。

其中各个模型下载：

[下载地址](https://www.ctolib.com/ymcui-Chinese-BERT-wwm.html)

## 脚本运行方式

### bert-wwm 数据预处理

若采用wwm方式bert预训练，则数据预处理运行脚本

```bash
python modi_create_pretraining_data.py \
--do_whole_word_mask=True \
--input_file=./data/test_seg.txt \
--output_file=./input/test_wwm.tf_record \
--vocab_file=./chinese_L-12_H-768_A-12/vocab.txt \
--do_lower_case=True \
--max_seq_length=512 \
--max_predictions_per_seq=20 \
--random_seed=12345 \
--dupe_factor=2 \
--masked_lm_prob=0.15 \
--short_seq_prob=0.1 \
```

其中`test_seg.txt`格式如下：

```text
美国总统特朗普4月14日在白宫新冠疫情简报会上宣布，他已指示美国政府暂停向世界卫生组织（WHO）拨款，称WHO在处理新冠疫情上不力，导致疫情在全球大爆发。
特朗普表示，美国政府非常关注对WHO的资助是否起到了效果。
美国政府正在推动一项针对世卫组织的调查，涵盖WHO“是否严重管理不当以及在掩盖新冠病毒传播方面所扮演的角色”。在调查期间暂停对WHO提供拨款。

中国工程院院士袁隆平提出最新指导意见：“不断发掘水稻耐盐碱基因，并将其转育到籼粳交高产杂交稻，特别是第三代杂交稻上。
”4月14日，在海南三亚崖州湾科技城，国家耐盐碱水稻技术创新中心试验现场观摩及建设推进会上，科技日报记者获取以上信息。
```

一行为一句，篇章之间分割采用行分割，空格为已经分好词语

### bert-base 数据预处理

```bash
python create_pretraining_data.py \
--do_whole_word_mask=False \
--input_file=./data/test.txt \
--output_file=./input/test.tf_record \
--vocab_file=./chinese_L-12_H-768_A-12/vocab.txt \
--do_lower_case=True \
--max_seq_length=512 \
--max_predictions_per_seq=20 \
--random_seed=12345 \
--dupe_factor=2 \
--masked_lm_prob=0.15 \
--short_seq_prob=0.1 \
```

其中`test_seg.txt`格式如下：


```text
美国总统特朗普4月14日在白宫新冠疫情简报会上宣布，他已指示美国政府暂停向世界卫生组织（WHO）拨款，称WHO在处理新冠疫情上不力，导致疫情在全球大爆发。
特朗普表示，美国政府非常关注对WHO的资助是否起到了效果。
美国政府正在推动一项针对世卫组织的调查，涵盖WHO“是否严重管理不当以及在掩盖新冠病毒传播方面所扮演的角色”。在调查期间暂停对WHO提供拨款。

中国工程院院士袁隆平提出最新指导意见：“不断发掘水稻耐盐碱基因，并将其转育到籼粳交高产杂交稻，特别是第三代杂交稻上。
”4月14日，在海南三亚崖州湾科技城，国家耐盐碱水稻技术创新中心试验现场观摩及建设推进会上，科技日报记者获取以上信息。
```

一行为一句，篇章之间分割采用行分割，采用的是中文字编码。

### bert 预训练

```bash
python pre_train.py \
--bert_config_file=./chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=./chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt \
--train_input_file=./input/train_wwm_trs.tf_record \
--dev_input_file=./input/dev_wwm_trs.tf_record \
--output_dir=./output \
--ckpt_path=./output/finetune \
--save_path=./output/model.ckpt \
--max_seq_length=512 \
--max_predictions_per_seq=20 \
--do_train=True \
--is_finetune=False \
--train_batch_size=6 \
--eval_batch_size=8 \
--learning_rate=5e-5 \
--num_train_epochs=3 \
--num_warmup_steps=10000 \
--save_checkpoints_steps=10000 \
--iterations_per_loop=10000 \
```
