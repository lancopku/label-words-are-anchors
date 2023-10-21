from icl.analysis.compress_top import compress, CompressTopArgs
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((CompressTopArgs,))
args, = parser.parse_args_into_dataclasses()
compress(args)