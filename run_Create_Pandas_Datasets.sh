grid_run --grid_mem=350G --grid_ncpus=6 \
  /apps/anaconda3/bin/jupyter nbconvert \
  --to notebook \
  --execute Create_Pandas_Datasets.ipynb \
  --output Create_Pandas_Datasets_executed.ipynb \
  --debug \
  --ExecutePreprocessor.timeout=-1
