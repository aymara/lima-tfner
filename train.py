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

from tfner.data_utils import CoNLLDataset
from tfner.ner_model import NERModel
from tfner.config import Config

import sys,argparse
from argparse import RawTextHelpFormatter


def main(language="eng"):
    # create instance of config
    try:
      config = Config(lang=language)
    except Exception as e:
      #print >> sys.stderr, "Exception: %s" % str
      print("Exception: %s" % e.args, file=sys.stderr)
      sys.exit(1)

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

    model.build_freeze()
    model.restore_session(config.dir_model)
    # freeze the graph
    model.freeze_my_graph()

    
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
