from icl.analysis.reweighting import train, ReweightingArgs
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((ReweightingArgs,))
args, = parser.parse_args_into_dataclasses()
train(args)