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

import numpy as np
import os
import re


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, pos, tag = ls[0],ls[1],ls[-1]
                    if(word.find("'s")==-1 or not(pos=="VBZ")):
                      if self.processing_word is not None:
                          word = self.processing_word(word)
                      if self.processing_tag is not None:
                          tag = self.processing_tag(tag)
                      words += [word]
                      tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def next_positions(i,j,length):
  if(j==length-1):
    j=0
    i+=1
  else:
    j+=1
  return i,j

def read_limafile(gold,filename_lima):
      accs = []
      correct_preds, total_correct, total_preds = 0., 0., 0.
      sequences,labels_pred=[],[]

      for (sequence,tags) in gold:
          sequences+=[sequence]
          labels_pred+=[tags]

      current_sentence,current_word=0,0
      pred=""
      label_pred=labels_pred[current_sentence][current_word]

      with open(filename_lima,"r") as lima:
        for line in lima:
          foundInPrevious= False
          line = line.strip()
          if (len(line) != 0 and not(line.startswith("-DOCSTART-"))):
              ls = line.split('\t')
              i,word,tag = int(ls[0]),ls[1],ls[5]
              word=word.split(" ")
              if(len(word)==1 and i!=0):
                if(sequences[current_sentence][current_word].find(word[0])!=0 and word[0].find("'s")==-1):
                  if(pred.find(word[0])==-1):
                    current_sentence,current_word=next_positions(current_sentence,current_word,len(sequences[current_sentence]))
                  else:
                    foundInPrevious=True
                if(foundInPrevious):
                  if(tag!="_"):
                    print(pred,label_pred,word[0],tag)
                    if(tag==label_pred):
                        correct_preds +=1
                    total_preds+=1
                  if(label_pred!="O"):
                    total_correct+=1
                else:
                  print(sequences[current_sentence][current_word],labels_pred[current_sentence][current_word],word[0],tag)
                  if(tag!="_"):
                    if(tag==labels_pred[current_sentence][current_word]):
                        correct_preds +=1
                    total_preds+=1
                  if(labels_pred[current_sentence][current_word]!="O"):
                    total_correct+=1
                  pred=sequences[current_sentence][current_word]
                  label_pred=labels_pred[current_sentence][current_word]
                  current_sentence,current_word=next_positions(current_sentence,current_word,len(sequences[current_sentence]))
              elif(len(word)>1 and i!=0):
                valid=False
                for i,w in enumerate(word):
                  foundInPrevious= False
                  if(i==0):
                    if(sequences[current_sentence][current_word].find(w)!=0 and w.find("'s")==-1):
                      if(pred.find(w)==-1):
                        current_sentence,current_word=next_positions(current_sentence,current_word,len(sequences[current_sentence]))
                      else:
                        foundInPrevious=True
                    if(foundInPrevious):
                      if(tag!="_"):
                        print(pred,label_pred,w,tag)
                        if(tag==label_pred):
                          valid=True
                        if(label_pred!="O"):
                          correct=True
                    else:
                      print(sequences[current_sentence][current_word],labels_pred[current_sentence][current_word],w,tag)
                      if(tag==labels_pred[current_sentence][current_word]):
                        valid=True
                      if(labels_pred[current_sentence][current_word]!="O"):
                        correct=True
                  else:
                    if(sequences[current_sentence][current_word].find(w)!=0 and w.find("'s")==-1):
                      if(pred.find(w)==-1):
                        current_sentence,current_word=next_positions(current_sentence,current_word,len(sequences[current_sentence]))
                      else:
                        foundInPrevious=True
                    #print(sequences[current_sentence][current_word],w,pred)
                    if(foundInPrevious):
                      print(sequences[current_sentence][current_word],labels_pred[current_sentence][current_word],w,tag)
                      if(label_pred.find(tag[2:]) and valid):
                          valid=True
                      elif(labels_pred[current_sentence][current_word]!="0"):
                          correct=True
                    else:
                      print(sequences[current_sentence][current_word],labels_pred[current_sentence][current_word],w,tag)
                      if(labels_pred[current_sentence][current_word].find(tag[2:]) and valid):
                        valid=True
                      elif(labels_pred[current_sentence][current_word]!="0"):
                          correct=True
                  if(not foundInPrevious):
                    pred=sequences[current_sentence][current_word]
                    label_pred=labels_pred[current_sentence][current_word]
                    current_sentence,current_word=next_positions(current_sentence,current_word,len(sequences[current_sentence]))
                total_preds+=1
                if(valid):
                  correct_preds +=1
                if(correct):
                  total_correct+=1

      p   = correct_preds / total_preds if correct_preds > 0 else 0
      r   = correct_preds / total_correct if correct_preds > 0 else 0
      f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0


      return {"f1": 100*f1, "precision" :100*p,"recall": 100*r}




def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def create_test_file(data,file):
    """

    """
    with open(file,"w") as f:
      y_batch = []
      for (x, y) in data:
        input_raw=""
        input_raw+=' '.join(x)+'\n'
        f.write(input_raw)
        y_batch += [y]
    return y_batch

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if int(tok) == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif int(tok) != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(int(tok), idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
