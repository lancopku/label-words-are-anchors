from icl.analysis.n_classification import n_classify, NClassificationArgs
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((NClassificationArgs,))
args, = parser.parse_args_into_dataclasses()
n_classify(args)