import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

imported = tf.saved_model.load("bertForPatent", tags="serve")
