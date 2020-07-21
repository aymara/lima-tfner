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

from model.data_utils import CoNLLDataset, read_limafile
from model.ner_model import NERModel
from model.config import Config
import sys,argparse
from argparse import RawTextHelpFormatter


def main(language,file_to_compare):
    # create instance of config
    try:
        config = Config(lang=language)
    except Exception as e:
        #print >> sys.stderr, "Exception: %s" % str
        print("Exception: %s" % e.args, file=sys.stderr)
        sys.exit(1)

    # build model
    model = NERModel(config)

    # create dataset
    gold  = CoNLLDataset(config.filename_test,None,None,None)

    metrics = read_limafile(gold,file_to_compare)

    msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])

    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''NE evaluation on dataset''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--lang', required=False, default="eng", help="Specify the language between french as fr and english as eng")
    parser.add_argument('--input_file', required=True, help="Specify the file that will be compared")
    try:
        arguments = parser.parse_args(args=sys.argv[1:])
        arguments=vars(arguments)
    except:
        parser.print_help()
        sys.exit(1)
    main(language=arguments["lang"],file_to_compare=arguments["input_file"])
