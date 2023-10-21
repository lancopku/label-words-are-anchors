from icl.analysis.compress import compress, CompressArgs
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((CompressArgs,))
args, = parser.parse_args_into_dataclasses()
compress(args)