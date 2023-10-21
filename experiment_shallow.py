from icl.utils.experiment_utils import Sh_run


model_names = ['gpt-j-6b']
seeds = [42,43,44,45,46]
task_names = ["emo", 'sst2', 'agnews', 'trec']
mask_layer_nums = [5,1,3,7]
hypers = [
    {
        'expr': 'python do_shallow_layer.py',
        # change to do_shallow_layer_non_label.py if you want to run that
        'task_name': task_names,
        "seed": seeds,
        'model_name': model_names,
        'mask_layer_num': mask_layer_nums,
        'mask_layer_pos': ['last', 'first'],
        'sample_size': [1000],
        'demonstration_shot':[1,2],
    },

]
run = Sh_run(hyperparameter_lists=hypers,
             gpu_list=[0,1,2,3,4,5])

run.run()
