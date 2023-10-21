from icl.utils.experiment_utils import Sh_run

model_names = ['gpt2-xl']
seeds = [42,43,44,45,46]
task_names = ['agnews', 'trec','emo','sst2']
hypers = [
    {
        'expr': 'python attention_attr.py',
        'task_name': task_names,
        "seed": seeds,
        'model_name': model_names,
        'sample_size': [1000],
        'demonstration_shot': [1]
    },
]
run = Sh_run(hyperparameter_lists=hypers,
             gpu_list=[0,1,3,4])

run.run()
