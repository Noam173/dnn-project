# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gc
import tensorflow as tf

gc.collect()
tf.keras.backend.clear_session()
tf.config.experimental.reset_memory_stats('GPU:0')