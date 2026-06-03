grid_run --grid_mem=350G --grid_ncpus=6 \
  /apps/anaconda3/bin/jupyter nbconvert \
  --to notebook \
  --execute Develop_coresignal_panel_runjing.ipynb \
  --output Develop_coresignal_panel_runjing_executed.ipynb \
  --debug \
  --ExecutePreprocessor.timeout=-1
