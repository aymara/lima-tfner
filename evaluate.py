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

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

import time, sys, argparse, re
from argparse import RawTextHelpFormatter

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")
        if words_raw == ["exit"]:
          break
        words_raw_formatted=[]
        str=""
        #seperate punctuations from text
        for word in words_raw:
          for char in word:
            if char.isalpha() or char.isdigit():
              str+=char
            else:
              if(str!=""):
                words_raw_formatted+=[str]
              words_raw_formatted+=[char]
              str=""
          if(str!=""):
            words_raw_formatted+=[str]
            str=""

        preds = model.predict(words_raw_formatted)
        to_print = align_data({"input": words_raw_formatted, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)

def main(language="eng"):
    #create instance of config
    try:
      config = Config(lang=language)
    except Exception as e:
      #print >> sys.stderr, "Exception: %s" % str
      print("Exception: %s" % e.args, file=sys.stderr)
      sys.exit(1)

    start_time = time.perf_counter()
    print("--- Execution time : %s seconds ---" % (time.perf_counter() - start_time))
    #build model
    start_time = time.perf_counter()
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    print("--- Execution time : %s seconds ---" % (time.perf_counter() - start_time))
    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''NE recognizer''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--lang', required=False, default="eng", help="Specify the language between french as fr and english as eng")
    try:
        arguments = parser.parse_args(args=sys.argv[1:])
        arguments=vars(arguments)
    except:
        parser.print_help()
        sys.exit(1)
    main(language=arguments["lang"])
