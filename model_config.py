#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author     Romaric Besançon (romaric.besancon@cea.fr)
# date       Tue Nov 10 2020
# copyright  Copyright (C) 2020 by CEA - LIST
#

import os,sys,argparse,codecs
import yaml
import jsonschema
from jsonschema import Draft4Validator, validators
import logging

from tfner import general_utils
from tfner.config import Config

class ConfigException(Exception):
    pass

def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property_, subschema in properties.items():
            if "default" in subschema and not isinstance(instance, list):
                if subschema["default"] == "null":
                    instance.setdefault(property_, None)
                else:
                    instance.setdefault(property_, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )
FillDefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)

ConfigSchema = {
  "$schema": "http://json-schema.org/schema#",
  "title": "Configuration for NER BiLSTM-CRF model",
  "type": "object",
  "properties": {
      "language": { "type": "string", "description": "language of the model."},
      "dir_output": { "type": "string", "description": "output directory.","default":"results"},
      "dim_word": { "type": "integer", "description": "dimension of the the word embeddings.","default":300},
      "dim_char": { "type": "integer", "description": "dimension of the the character embeddings.","default":100},
      "filename_glove": { "type": "string", "description": "path to glove embeddings"},
      "filename_trimmed": { "type": ["string","null"], "description": "path to trimmed embeddings (created)","default":"null"},
      "use_pretrained": { "type": "boolean", "description": "indicates if pretrained embeddings are used","default":True},
      "filename_dev": { "type": "string", "description": "path to IOB file containing training data"},
      "filename_train": { "type": "string", "description": "path to IOB file containing training data"},
      "filename_test": { "type": "string", "description": "path to IOB file containing training data"},

      # vocab (created from dataset with build_data.py)
      "filename_words": { "type": "string", "description": "path to output file containing training words", "default":"data/IOB1/eng/words.txt"},
      "filename_tags": { "type": "string", "description": "path to output file containing training tags", "default":"data/IOB1/eng/tags.txt"},
      "filename_chars": { "type": "string", "description": "path to output file containing training chars", "default":"data/IOB1/eng/chars.txt"},

      "dir_resources": { "type": "string", "description": "data directory", "default":"data/IOB1/eng/"},
      "dir_model": { "type": "string", "description": "name of output file containing the model weights (relative to dir_output)", "default":"model.weights"},
      "path_log": { "type": "string", "description": "name of output file containing the log (relative to dir_output)", "default":"log.txt"},
      "input_graph": { "type": "string", "description": "path to output file containing training chars (relative to dir_output)", "default":"input_graph.pb"},
      "output_graph": { "type": "string", "description": "path to output file containing training chars (relative to dir_output)", "default":"output_graph.pb" },
      
      # parameters for the NN model and training
      "max_iter": { "type": ["string","null"], "description": "if not null, max number of examples in Dataset", "default":"null" },
      "train_embeddings": { "type": "boolean", "description": "indicates if pretrained embeddings are used","default":False},
      "nepochs": { "type": "integer", "description": "max number of epochs","default":100},
      "dropout": { "type": "number", "description": "dropout probability","default":0.5},
      "batch_size": { "type": "integer", "description": "batch size","default":20},
      "lr_method": { "type": "string", "description": "optimizer","default":"adam"},
      "lr": { "type": "number", "description": "learning rate","default":0.001},
      "lr_decay": { "type": "number", "description": "learning rate decay","default":0.9},
      "clip": { "type": "number", "description": "if negative, no clipping","default":-1},
      "nepoch_no_imprv": { "type": "integer", "description": "nb epochs without improvement","default":3},
      "hidden_size_char": { "type": "integer", "description": "size of lstm on chars","default":100},
      "hidden_size_lstm": { "type": "integer", "description": "size of lstm on word embeddings","default":300},
  
      "use_crf": { "type": "boolean", "description": "use the CRF","default":True},
      "use_chars": { "type": "boolean", "description": "use character representations","default":True}
  }
}

class ModelConfig(Config):
    def __init__(self,config_file=None,config_data=None):
        if config_file is not None:
            try:
                with open(config_file) as f:
                    # use safe_load instead load
                    cfg = yaml.load(f,Loader=yaml.Loader)
                    ok,err=self.validate(cfg)
                    if not ok:
                        msg="Error in config file: %s"%err
                        logging.error(msg)
                        raise ConfigException(msg)
                    self.__dict__.update(**cfg)
            except IOError:
                logging.error("Failed to open file %s"%f)
                raise ConfigException("Failed to open file %s"%f)
            except yaml.parser.ParserError as e:
                logging.error("Failed to parse file %s: %s"%(f,str(e)))
                raise ConfigException(str(e))

        if config_data is not None:
            # copy data to validate and set default
            cfg=copy.copy(config_data)
            ok,err=self.validate(cfg)
            if not ok:
                msg="Error in configuration data: %s"%err
                logging.error(msg)
                raise ConfigException(msg)
            self.__dict__.update(**cfg)

        # validate paths
        self.filename_glove=self.validate_path(self.filename_glove)
        if not self.filename_trimmed:
            self.filename_trimmed=self.filename_glove+".trimmed.npz"

        # paths relative to output directory
        self.dir_output=self.validate_path(self.dir_output,create=True)
        self.dir_model=os.path.join(self.dir_output,self.dir_model)
        self.path_log=os.path.join(self.dir_output,self.path_log)
        self.input_graph=os.path.join(self.dir_output,self.input_graph)
        self.output_graph=os.path.join(self.dir_output,self.output_graph)

        # create instance of logger
        self.logger = general_utils.get_logger(self.path_log)


    def validate(self,data):
        try:
            #jsonschema.validate(instance=data, schema=ConfigSchema)
            FillDefaultValidatingDraft4Validator(ConfigSchema).validate(data)
        except jsonschema.exceptions.ValidationError as err:
            logging.error(err)
            return False,err.message
        return True,""
        
    def validate_path(self,path,create=False):
        if path is not None:
            #if self.working_dir is not None:
            #    path=os.path.join(self.working_dir,path)
            # expand environment variables
            path=os.path.expandvars(path)
            if not os.path.exists(path):
                if create:
                    os.makedirs(path)
                    return path
                else:
                    logging.error("path does not exist: %s"%path)
                    return None
        return path
