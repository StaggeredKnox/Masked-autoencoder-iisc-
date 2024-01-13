POINTS TO NOTE :
----------------

timm folders (timm_0.25, ..) contain hf transformer models and not timm models.

for running continue_training.py--->
    in config_fbMae.py use cmd at [line-70] (for fbMae_model)
    in config_timm.py use cmd at [line-90] (for timm_mae_model)

for running GPUs_fbMae_finetuning.py--->
    in config_GPUs_fbMae.py use cmd at [line-5]

GPUs_fbMae_finetuning.py has continue_training mechanism in_built,
so in case training stops simply go to its configuration file and
turn "load_latest" key to True.

all models while training only log training images to wandb and not validation images,
however, test images get to display only after the complete training.

one could use testing.py file to get more insights about the quality of training on test images,
but "test_on_datapath" key must be set aptly.

needless to say that we are not using self implemented mae_model so none of the above features like,
continue_training, sweep and the rest could not be guaranteed to run.

Except above everything seems sorted and running.

____________________________________________________________________________________________________________
-THANKS :)

# Test data split from main data and evaluate on that data
# log validation set 
# fb mae - in reconwtruct plots function, we are copying unmasked patches to reconstruced plot
# add random seed in random split functions and config