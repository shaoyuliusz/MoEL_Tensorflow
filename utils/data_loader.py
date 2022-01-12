import sys
import os
from numpy import int64

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
import random
import math
import os
import logging
from utils import config
import pickle
from tqdm import tqdm
import pprint

pp = pprint.PrettyPrinter(indent=1)
import re
import ast

import time
from model.common_layer import write_config
from utils.data_reader import load_dataset
from tensorflow.python.ops import array_ops

tf.random.set_seed(1)


pairs_tra, pairs_val, pairs_tst, vocab = load_dataset(print_sample=False)
# remain the same data structure and variable namings as the same as the original paper

# this creates list of dict
def process_item(data):
    """
    data: 4-key dictionary of lists. Returns a dictionary of lists.

    Ref to original MoEL paper.
    """
    emo_map = {
        "surprised": 0,
        "excited": 1,
        "annoyed": 2,
        "proud": 3,
        "angry": 4,
        "sad": 5,
        "grateful": 6,
        "lonely": 7,
        "impressed": 8,
        "afraid": 9,
        "disgusted": 10,
        "confident": 11,
        "terrified": 12,
        "hopeful": 13,
        "anxious": 14,
        "disappointed": 15,
        "joyful": 16,
        "prepared": 17,
        "guilty": 18,
        "furious": 19,
        "nostalgic": 20,
        "jealous": 21,
        "anticipating": 22,
        "embarrassed": 23,
        "content": 24,
        "devastated": 25,
        "sentimental": 26,
        "caring": 27,
        "trusting": 28,
        "ashamed": 29,
        "apprehensive": 30,
        "faithful": 31,
    }
    ls_item = []
    for index in range(0, len(data["context"])):

        item = {}
        item["context_text"] = data["context"][index]
        item["target_text"] = data["target"][index]
        item["emotion_text"] = data["emotion"][index]

        item["context"], item["context_mask"] = preprocess(item["context_text"])

        item["target"] = preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = preprocess_emo(
            item["emotion_text"], emo_map
        )
        ls_item.append(item)

    ls_item.sort(key=lambda x: len(x["context"]), reverse=True)

    item_info = {}  # item_info is a 8 key dictionary of lists
    for key in ls_item[0].keys():
        item_info[key] = [d[key] for d in ls_item]
    return item_info  # dict of lists


def preprocess(arr, anw=False):
    """Converts words to ids."""
    if anw:
        sequence = [
            vocab.word2index[word] if word in vocab.word2index else config.UNK_idx
            for word in arr
        ] + [config.EOS_idx]
        return sequence
    else:
        X_dial = [config.CLS_idx]
        X_mask = [config.CLS_idx]
    for i, sentence in enumerate(arr):
        X_dial += [
            vocab.word2index[word] if word in vocab.word2index else config.UNK_idx
            for word in sentence[0]
        ]  # add a 0
        spk = vocab.word2index["USR"] if i % 2 == 0 else vocab.word2index["SYS"]
        X_mask += [spk for _ in range(len(sentence[0]))]  # add a 0  [[]]
    assert len(X_dial) == len(X_mask)

    return X_dial, X_mask


def preprocess_emo(emotion, emo_map):
    """Ref to original MoEL paper"""
    program = [0] * len(emo_map)
    program[emo_map[emotion]] = 1

    return program, emo_map[emotion]

<<<<<<< HEAD
   
#each item has 7 tensors
def convert_to_dataset(item_info,batch_size = 32):
    """convert to Tensorflow data.DataSet format. Based on and modified from: 
=======

# each item has 7 tensors
def convert_to_dataset(item_info, batch_size=32):
    """convert to Tensorflow data.DataSet format. Ref:
>>>>>>> d985bf71f2a61c95c8a9ee08ae3996bea5e9229b
    https://medium.com/analytics-vidhya/tutorial-on-bucket-by-sequence-length-api-for-efficiently-batching-nlp-data-while-training-20d8ef5219d7"""

    def _element_length_fn(a, b, c, d, e, f, g):
        return array_ops.shape(a)[0]

    def generator():
        # define a generator to tensorflow to create the dataset
        for i in range(0, len(item_info["context"])):
            input_batch = item_info["context"][i]  # var len [...]
            input_length = [len(item_info["context"])]  # [int]
            mask_input = item_info["context_mask"][i]  # var len [...]
            target_batch = item_info["target"][i]  # var len [...]
            target_length = [len(item_info["target"][i])]  # [int]
            target_program = item_info["emotion"][i]  # var len [...]
            program_label = [item_info["emotion_label"][i]]  # [int]
            yield input_batch, input_length, mask_input, target_batch, target_length, target_program, program_label

    # create tf.data.Dataset object from generator.
    toy_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=([None]), dtype=tf.int32),
            tf.TensorSpec(shape=([1]), dtype=tf.int32),
            tf.TensorSpec(shape=([None]), dtype=tf.int32),
            tf.TensorSpec(shape=([None]), dtype=tf.int32),
            tf.TensorSpec(shape=([1]), dtype=tf.int32),
            tf.TensorSpec(shape=([None]), dtype=tf.int32),
            tf.TensorSpec(shape=([1]), dtype=tf.int32),
        ),
    )
    bound_num = len(item_info["context"]) // batch_size

    # let's use context length for bucket_by_sequence_length
    toy_boundaries = [
        len(x) for i, x in enumerate(item_info["context"]) if i % bound_num == 0
    ]
    # [batch_size] * len(bucket_boundaries) + 1
    toy_bucket_batch_sizes = [batch_size] * (len(toy_boundaries) + 1)

    # this is similar to the "merge" function to pad lists for each batch in the original code
    dataset = toy_dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            _element_length_fn,
            bucket_boundaries=toy_boundaries,
            bucket_batch_sizes=toy_bucket_batch_sizes,
            drop_remainder=True,
            # padding_values=1,
            pad_to_bucket_boundary=False,
        )
    )
    return dataset


def prepare_data_seq(batch_size):
    """a wrapper to generate batched data ready for training/testing"""
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    emo_map = {
        "surprised": 0,
        "excited": 1,
        "annoyed": 2,
        "proud": 3,
        "angry": 4,
        "sad": 5,
        "grateful": 6,
        "lonely": 7,
        "impressed": 8,
        "afraid": 9,
        "disgusted": 10,
        "confident": 11,
        "terrified": 12,
        "hopeful": 13,
        "anxious": 14,
        "disappointed": 15,
        "joyful": 16,
        "prepared": 17,
        "guilty": 18,
        "furious": 19,
        "nostalgic": 20,
        "jealous": 21,
        "anticipating": 22,
        "embarrassed": 23,
        "content": 24,
        "devastated": 25,
        "sentimental": 26,
        "caring": 27,
        "trusting": 28,
        "ashamed": 29,
        "apprehensive": 30,
        "faithful": 31,
    }

    # call process_item on train, valid, and test dicts
    item_info_tra = process_item(pairs_tra)
    item_info_val = process_item(pairs_val)
    item_info_tst = process_item(pairs_tst)

    # convert to tf.data.DataSet format
    data_loader_tra = convert_to_dataset(item_info_tra)
    data_loader_val = convert_to_dataset(item_info_val)
    data_loader_tst = convert_to_dataset(item_info_tst)

    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(emo_map)
