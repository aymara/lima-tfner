#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author     Romaric Besançon (romaric.besancon@cea.fr)
# date       Tue Dec  1 2020
# copyright  Copyright (C) 2020 by CEA - LIST
#

import sys,argparse,codecs

from tfner.configfromfile import ModelConfig
from tfner.core import eval_model

#----------------------------------------------------------------------
# main function
def main(argv):
    # parse command-line arguments
    parser=argparse.ArgumentParser(description="eval the model specified in the configuration file on the test file also specified in the configuration file")
    # optional arguments
    parser.add_argument("--print_results",action="store_true",help="print the predicted labels: one word per line as a pair (ref label,predicted label)")
    parser.add_argument("--cpp_api",action="store_true",help="use the C++ API")
    # positional arguments
    parser.add_argument("config_file",help="configuration file indicating the parameters for the training")
    
    param=parser.parse_args()

    # do main
    config=ModelConfig(param.config_file)
    eval_model(config, print_results=param.print_results, use_cpp_api=param.cpp_api)

if __name__ == "__main__":
    main(sys.argv[1:])
