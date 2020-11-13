__license__ = """
 Copyright (C) 2017 Guillaume Genthial
 Modifications copyright (C) 2020 CEA LIST

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True,lang='eng'):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.language=lang

        if(self.language=='eng'):
          # outputs
          self.dir_output = 'results/eng_IOB1/'

          # embeddings
          self.dim_word = 300
          self.dim_char = 100

          # glove files
          self.filename_glove = "/home/romaric/data/embeddings/glove/glove.6B.{}d.txt".format(self.dim_word)

          # trimmed embeddings (created from glove_filename with build_data.py)
          self.filename_trimmed = "/home/romaric/data/embeddings/glove/glove.6B.{}d.trimmed.npz".format(self.dim_word)
          self.use_pretrained = True

          self.dir_resources='data/IOB1/eng/'

          # dataset
          self.filename_dev = "data/IOB1/eng/eng.testa"
          self.filename_test = "data/IOB1/eng/eng.testb"
          #self.filename_test = "./data/tests_btw_two_process_units/Wiki-eng-37/corpusEngWikiNER.txt.2"
          self.filename_train = "data/IOB1/eng/eng.train"

          #filename_dev = filename_test = filename_train = "data/test.txt" # test

          # vocab (created from dataset with build_data.py)
          self.filename_words = "data/IOB1/eng/words.txt"
          self.filename_tags = "data/IOB1/eng/tags.txt"
          self.filename_chars = "data/IOB1/eng/chars.txt"
        elif(self.language=='fr'):
          # outputs
          self.dir_output = 'results/fr/'

          # embeddings
          self.dim_word = 300
          self.dim_char = 100

          # glove files
          self.filename_glove = "data/word2vecFR/word2vecFR.vec".format(self.dim_word)

          # trimmed embeddings (created from glove_filename with build_data.py)
          self.filename_trimmed = "data/word2vecFR.trimmed.npz".format(self.dim_word)
          self.use_pretrained = True

          self.dir_resources="data/IOB1/fr/"

          # dataset
          self.filename_dev = "data/IOB1/fr/fr.testa"
          #self.filename_test = "data/IOB1/fr/fr.testb"
          self.filename_test = "data/corpusFrWikiNER.txt.2"
          self.filename_train = "data/IOB1/fr/fr.train"

          #filename_dev = filename_test = filename_train = "data/test.txt" # test

          # vocab (created from dataset with build_data.py)
          self.filename_words = "data/IOB1/fr/words.txt"
          self.filename_tags = "data/IOB1/fr/tags.txt"
          self.filename_chars = "data/IOB1/fr/chars.txt"
        else:
          raise Exception("This language is not supported. "\
                                    "Only French and English are supported.")

        self.dir_model  = self.dir_output + "model.weights/"
        self.path_log   = self.dir_output + "log.txt"
        self.input_graph=self.dir_output + 'input_graph.pb'
        self.output_graph=self.dir_output + 'output_graph.pb'

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

    #general config

    max_iter = None # if not None, max number of examples in Dataset

    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
