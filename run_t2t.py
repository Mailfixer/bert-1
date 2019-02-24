# this file is based on tensor2tensor/bin/t2t-trainer
"""Trainer for Tensor2Tensor.
This script is used to train your models in Tensor2Tensor.
For example, to train a shake-shake model on MNIST run this:
t2t-trainer \
  --generate_data \
  --problem=image_mnist \
  --data_dir=~/t2t_data \
  --tmp_dir=~/t2t_data/tmp
  --model=shake_shake \
  --hparams_set=shake_shake_quick \
  --output_dir=~/t2t_train/mnist1 \
  --train_steps=1000 \
  --eval_steps=100
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
print(tf.__version__)

from tensor2tensor.bin import t2t_trainer


"""
run t2t experiment for translation
add the model that uses bert (register to tensor2tensor's model collection)
add the eng-eng data
"""


def main(argv):
  t2t_trainer.main(argv)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()


