


grid_run --grid_mem=350G --grid_ncpus=6 \
  /apps/anaconda3/bin/jupyter nbconvert \
  --to notebook \
  --execute Match_to_SCP_data.ipynb \
  --output Match_to_SCP_data_executed.ipynb \
  --debug \
  --ExecutePreprocessor.timeout=-1
