from icl.util_classes.arg_classes import ShallowNonLabelArgs
from icl.analysis.shallow_layer_non_label import shallow_layer_non_label
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((ShallowNonLabelArgs,))
args, = parser.parse_args_into_dataclasses()
shallow_layer_non_label(args)