#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author     Romaric Besançon (romaric.besancon@cea.fr)
# date       Tue Nov 10 2020
# copyright  Copyright (C) 2020 by CEA - LIST
#

import sys,argparse,codecs

import tfner.data_utils as utils
from tfner.ner_model import NERModel

#----------------------------------------------------------------------
def build_data(config):

    processing_word = utils.get_processing_word(lowercase=True)

    # Generators
    dev   = utils.CoNLLDataset(config.filename_dev, processing_word)
    test  = utils.CoNLLDataset(config.filename_test, processing_word)
    train = utils.CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = utils.get_vocabs([train, dev, test])
    vocab_glove = utils.get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(utils.UNK)
    vocab.add(utils.NUM)

    # Save vocab
    utils.write_vocab(vocab, config.filename_words)
    utils.write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = utils.load_vocab(config.filename_words)
    utils.export_trimmed_glove_vectors(vocab, config.filename_glove,
                                       config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = utils.CoNLLDataset(config.filename_train)
    vocab_chars = utils.get_char_vocab(train)
    utils.write_vocab(vocab_chars, config.filename_chars)


def build_model(config):

    # preprocess data
    build_data(config)
    
    # load embeddings
    config.load()
    
    # build model
    model = NERModel(config)
    model.build()
    dev   = utils.CoNLLDataset(config.filename_dev, config.processing_word,
                               config.processing_tag, config.max_iter)
    train = utils.CoNLLDataset(config.filename_train, config.processing_word,
                               config.processing_tag, config.max_iter)
    model.train(train, dev)

    # save model
    model.build_freeze()
    model.restore_session(config.dir_model)
    model.freeze_my_graph()
    
def eval_model(config, print_results=False, use_cpp_api=False):
    # suppose the data is already preprocessed (build_model should have been called before the eval)
    config.load()
    model = NERModel(config)
    # evaluate and interact
    if use_cpp_api:
        test  = utils.CoNLLDataset(config.filename_test, None,
                                   config.processing_tag, config.max_iter)
        model.evaluate_on_cplusplus_api(test,print_results=print_results)
    else:
        model.build()
        model.restore_session(config.dir_model)
        test  = utils.CoNLLDataset(config.filename_test, config.processing_word,
                                   config.processing_tag, config.max_iter)
        model.evaluate(test,print_results=print_results)
        

    
