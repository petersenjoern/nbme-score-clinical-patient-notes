import os
import random
import numpy as np
import tensorflow as tf


def seed_env(seed: int):
    """seed various services & libraries"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
