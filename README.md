# Named Entity Recognition with Tensorflow

This repo implements a NER model using Tensorflow (LSTM + CRF + chars embeddings). It includes modifications by CEA LIST for its use inside LIMA.

Versions : Tensoflow 1.3 and Python 3.5

Check the [blog post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```


## Model

Similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF



## Getting started

We need to download a dataset in any language you want and pre-trained embeddings (not necessarily).

For English, GloVe embeddings are the best.

To download the GloVe vectors :

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.

To build the training data, train and evaluate the model with

```
make run
```


## Details

Here is the breakdown of the commands executed in `make run`:

The language used by default is English.

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```bash
python build_data.py --lang eng/fr
```

2. Train the model with

```bash
python train.py --lang eng/fr
```

3. Evaluate and interact with the model with

```bash
python evaluate.py --lang eng/fr
```

4. Export the model with

```bash
python freezeGraph.py --lang eng/fr
```

5. Evaluate with C++ API

```bash
python testAPIC++.py --lang eng/fr
```

Data iterators and utils are in model/data_utils.py and the model with training/test procedures is in model/ner_model.py

## Training Data

The training data must be in the following format named IOB (Inside-Output-Beginning), identical to the CoNLL'03 dataset [1].

It is recommended to use IOBES annotations, performances are better with this format.

State-of-the-art performance (F1 score between 90 and 91) have been reached training on the English corpora CoNLL'03.

Results with French and English are avalaible in **results** folder.

I use English CoNLL'03 dataset + GloVe embeddings (d-300) and WikiNER [2] dataset (aij-wikiner-fr-wp3) + fastText embeddings (d-300) [3] for French.

In `config.py`, all are specified.


Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
dev_filename = "data/coNLL/eng/eng.testa.iob"
test_filename = "data/coNLL/eng/eng.testb.iob"
train_filename = "data/coNLL/eng/eng.train.iob"
```

If you want to use a new language, you have to precise it in  `config.py` like

```python
if(self.language=='yourlanguage'):
    # outputs
    self.dir_output = '...'

    # embeddings
    self.dim_word = '...'
    self.dim_char = '...'

    # embeddings files
    self.filename_glove = "...".format(self.dim_word)

    # trimmed embeddings (created from glove_filename with build_data.py)
    self.filename_trimmed = "...".format(self.dim_word)
    self.use_pretrained = True/False

    self.dir_resources="data/format_used/yourlanguage/" format_used={IOB1,IOB2,IOBES,...}

    # dataset
    self.filename_dev = "data/format_used/yourlanguage/yourlanguage.testa"
    self.filename_test = "data/format_used/yourlanguage/yourlanguage.testb"
    self.filename_train = "data/format_used/yourlanguage/yourlanguage.train"

    # vocab (created from dataset with build_data.py)
    self.filename_words = "data/format_used/yourlanguage/words.txt"
    self.filename_tags = "data/format_used/yourlanguage/tags.txt"
    self.filename_chars = "data/format_used/yourlanguage/chars.txt"
```

To create French corpora based on WikiNER, you need to follow these instructions :

```
data/system2conll.pl aijwikinerenwp3.bz2
python parse_fr_data.py
```

To create English corpora, you need to follow the README provided by coNLL folder.

To compare the old process unit from the new one, you can use the script Python compare.py

To install files in LIMA you have to precise the paths in apic++/moduleNER/CMakeLists.txt


## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.



## Citations

[1]   Sang, Erik F. Tjong Kim, et Fien De Meulder. « Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition ». *arXiv:cs/0306050*, 12 juin 2003. <http://arxiv.org/abs/cs/0306050>.

[2]  Nothman, Joe, Nicky Ringland, et Will Radford. « Learning multilingual named entity recognition from Wikipedia », 2012. <https://ac.els-cdn.com/S0004370212000276/1-s2.0-S0004370212000276-main.pdf?_tid=d2d8c47c-e003-11e7-a9d8-00000aab0f6b&acdnat=1513169448_9f6447041adb80408317a8d2019ed899>.

[3] <https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md>
