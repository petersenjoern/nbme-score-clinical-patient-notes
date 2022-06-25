import os
import tensorflow as tf

#TODO: type annotation
#TODO: add logging
def define_gpu_strategy():
    """Define the GPU strategy for modelling."""

    gpu_available = tf.config.list_physical_devices('GPU')

    if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.MirroredStrategy()
        
    return strategy

#TODO: add logging
def set_mixed_precision():
    """Set TF mixed precision."""
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

