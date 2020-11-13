from tfner.data_utils import CoNLLDataset, create_test_file
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

    # create dataset
    test  = CoNLLDataset(config.filename_test,None,config.processing_tag,config.max_iter)

    # evaluate and interact
    model.evaluate_on_cplusplus_api(test) # call ner_model.evaluate 
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''NE evaluation on dataset''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--lang', required=False, default="eng", help="Specify the language between french as fr and english as eng")
    try:
        arguments = parser.parse_args(args=sys.argv[1:])
        arguments=vars(arguments)
    except:
        parser.print_help()
        sys.exit(1)
    main(language=arguments["lang"])
