
common_config = {
    'data_dir': '/content/crnn-pytorch/data/',
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 100,
    'train_batch_size': 8,
    'eval_batch_size': 512,
    'lr': 0.0001,
    'dropout': 0.5,
    'show_interval': 3000,
    'valid_interval': 6000,
    'save_interval': 6000,
    'cpu_workers': 2,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': '/content/drive/MyDrive/Colab Notebooks/output/crnn-pytorch8/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 2,
    'reload_checkpoint': 'checkpoints/crnn_synth90k.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
