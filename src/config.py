
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
    'train_batch_size': 32,
    'eval_batch_size': 512,
    'lr': 0.001,
    'weight_decay': 0.00001,
    'dropout': 0.5,
    'show_interval': 2000,
    'valid_interval': 4000,
    'save_interval': 4000,
    'cpu_workers': 2,
    'reload_checkpoint': '/content/drive/MyDrive/Colab Notebooks/resources/crnn_104000_loss0.20359827135129926_cer0.017572328448295593.pt',
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
