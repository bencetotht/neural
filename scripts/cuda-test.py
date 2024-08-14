import torch
import tensorflow as tf
print(f'Torch detected CUDA: {torch.cuda.is_available()}')
print(f'Tensorflow detected GPUs: {len(tf.config.list_physical_devices("GPU"))}')