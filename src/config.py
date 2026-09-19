
common_config = {
    'data_dir': '/content/crnn-pytorch/data/',
    'img_width': 160,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 24,
    'train_batch_size': 128,
    'eval_batch_size': 512,
    # AdamW with a linear warmup over 'warmup_iterations' followed by cosine decay to 'lr_min' over all epochs.
    'lr': 0.0003,
    'lr_min': 0.000001,
    'warmup_iterations': 500,
    'weight_decay': 0.01,
    'dropout': 0.3,
    # Online augmentation of the training crops (see augment.py). Set to None to train on the plain crops.
    'augmentation': {
        'scale': (0.8, 1.15),          # text size inside the cell: < 1 adds margin, > 1 crops the borders
        'rotate': 3.0,                 # degrees
        'shear': 0.15,
        'thicken_probability': 0.25,   # min filter: thicker strokes
        'thin_probability': 0.15,      # max filter: thinner strokes
        'contrast': (0.75, 1.25),
        'brightness': 0.15,            # fraction of the grey range
        'blur_probability': 0.3,
        'blur_radius': (0.3, 1.0),
        'noise_probability': 0.3,
        'noise_sigma': 0.04,           # fraction of the grey range
        'offset_probability': 0.5,     # random horizontal placement on the canvas instead of left-aligned
    },
    # Sampling weights: crops of under-represented collections and labels with rare characters are drawn more often.
    'collection_weights': {'othr': 2.0, 'my': 1.5, 'ajcc': 1.0, 'hcs': 1.0},
    'rare_char_threshold': 2000,
    'rare_char_weight': 3.0,
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
    'reload_checkpoint': 'results/crnn-pytorch-css-22/crnn_032000_loss0.69584_cer0.05218_acc0.90302.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
    # Report the n most probable sequences per crop (oracle accuracy and confidence calibration); None for top-1 only.
    'n_best': 3,
}
evaluate_config.update(common_config)
