"""随机种子设置工具，供训练与评估脚本复用。"""
import os
import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """固定 Python / NumPy / TensorFlow 随机种子，尽量保证实验可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
