## README

### Preparations

You can use your own models by adding their folder path to ROOT_PATH_LIST in './icl/utils/load_local.py'. Otherwise, they will be downloaded from Huggingface.

Datasets will be downloaded from Huggingface when you run the code.



Dependencies are listed in 'requirements.txt'.



### Validation of the Hypothesis (Section 2)

#### Section 2.1

Run './attention_attr.py' to get the results of $S_{wp},S_{pq},S_{ww}$.

To run all experiments, you can use 'experiment_attn_attr.py' to generate the shell file 'gpu_sh.sh'.

Use './attention_attr_ana.ipynb' to analyze the result.



#### Section 2.2

Run 'do_shallow_layer.py' or generate the shell file via 'experiment_shallow.py'

Use './shallow_analysis.ipynb' to analyze the result.



#### Section 2.3

Run 'do_deep_layer.py' or generate the shell file via 'experiment_deep.py'

Use './deep_analysis.ipynb' to analyze the result.



### Applications of the Hypothesis (Section 3)

#### Section 3.1

Run 'reweighting.py' or generate the shell file via 'experiment_reweighting.py'

Use './reweighting.ipynb' to analyze the result.



#### Section 3.2

Run 'do_compress.py' or generate the shell file via 'experiment_compress.py' (to use $Hidden_{random-top}$ instead of $Hidden_{random}$, run 'do_compress_top.py')

Use 'compress_analysis.ipynb' to analyze the result.

To record the used time of the method, run 'do_compress_time.py'



#### Section 3.3

Use 'Error_analysis.ipynb' to run the experiment.



### Other results

#### Vanilla ICL Results

Run 'do_nclassify.py' or generate the shell file via 'experiment_ncls.py'

Use 'nclassfication.ipynb' to read the results.





