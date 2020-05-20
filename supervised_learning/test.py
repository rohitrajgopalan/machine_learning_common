from classifier_experiments import *
from harmony_ml import *

drop_cols_for_each_dataset()
prepare_each_dataset_for_classification('NUM_COLLISIONS')
perform_and_plot_experiment_on_data(df_from_each_file)
