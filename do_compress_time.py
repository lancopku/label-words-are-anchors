from icl.analysis.compress_time import compress_time, CompressTimeArgs
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((CompressTimeArgs,))
args, = parser.parse_args_into_dataclasses()
compress_time(args)