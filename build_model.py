#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author     Romaric Besançon (romaric.besancon@cea.fr)
# date       Tue Nov 10 2020
# copyright  Copyright (C) 2020 by CEA - LIST
#

import sys,argparse,codecs

from tfner.configfromfile import ModelConfig
from tfner.core import build_model
    
#----------------------------------------------------------------------
# main function
def main(argv):
    # parse command-line arguments
    parser=argparse.ArgumentParser(description="build a NN model for NER")
    # optional arguments
    #parser.add_argument("--arg",type=int,default=42,help="description")
    # positional arguments
    parser.add_argument("config_file",help="configuration file indicating the parameters for the training")
    
    param=parser.parse_args()

    # do main
    cfg=ModelConfig(param.config_file)
    build_model(cfg)

if __name__ == "__main__":
    main(sys.argv[1:])
