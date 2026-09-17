
common_config = {
    'data_dir': '/content/crnn-pytorch/data/',
    'img_width': 160,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 10,
    'train_batch_size': 128,
    'eval_batch_size': 512,
    # AdamW with a linear warmup over 'warmup_iterations' followed by cosine decay to 'lr_min' over all epochs.
    'lr': 0.0003,
    'lr_min': 0.000001,
    'warmup_iterations': 500,
    'weight_decay': 0.01,
    'dropout': 0.5,
    'show_interval': 2000,
    'valid_interval': 4000,
    'save_interval': 4000,
    'cpu_workers': 2,
    # Decoded crops are cached as '<cache_dir>/<split>_<h>x<w>.npy'; keep it on the local Colab disk for fast access.
    'cache_dir': '/content/crnn-pytorch/data/',
    # bfloat16 on L4/A100, float16 with loss scaling on T4; set to False to train in float32.
    'mixed_precision': True,
    'reload_checkpoint': None,
    # Optional 'crnn_train_state.pt' written next to the checkpoints to resume optimizer state and iteration counter.
    'reload_train_state': None,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': '/content/drive/MyDrive/Colab Notebooks/output/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 2,
    'reload_checkpoint': 'results/crnn-pytorch-css-17/crnn_468000_loss0.21019311518523184_acc0.9585422790938959.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
