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

from model.config import Config

import sys,argparse
from argparse import RawTextHelpFormatter

from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main(language="eng"):
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    try:
      config = Config(load=False,lang=language)
    except Exception as e:
      #print >> sys.stderr, "Exception: %s" % str
      print("Exception: %s" % e.args, file=sys.stderr)
      sys.exit(1)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''NE recognizer''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--lang', required=False, default="eng", help="Specify the language between french as fr and english as eng")
    try:
        arguments = parser.parse_args(args=sys.argv[1:])
        arguments=vars(arguments)
        print(arguments["lang"])
    except:
        parser.print_help()
        sys.exit(1)
    main(language=arguments["lang"])
