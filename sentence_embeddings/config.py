"""
Configuration file.
"""
import os

thought_size = 480
embed_dim = 256
vocab_size = 2000
max_length = 42
epochs = 50
# lr = 5e-4
total_sent = 2000
batch_size_per_gpu = 128/4
validation_size = 0.3
learning_rate = 0.001,
learning_rate_decay_factor = 0.5,
learning_rate_decay_steps = 400000,
number_of_steps = 500000,
clip_gradient_norm = 5.0

checkpoint_dir = '/scratch/smuthi2s/NLP_data/logs_480'
checkpoint_prefix = os.path.join(checkpoint_dir, "2000_ckpt")
