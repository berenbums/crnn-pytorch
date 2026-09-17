import os
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from dataset import CssDataset, css_collate_fn
from model import CRNN
from evaluate import evaluate, format_evaluation
from config import train_config as config


def train_batch(crnn, data, optimizer, scheduler, criterion, device, scaler, amp_dtype):
    crnn.train()
    images, targets, target_lengths = [d.to(device, non_blocking=True) for d in data]

    # Mixed precision for the CNN/LSTM forward pass; the CTC loss is always computed in float32.
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def select_amp_dtype(device):
    """Pick the autocast dtype: bfloat16 where the GPU supports it (L4, A100), float16 otherwise (T4), none on CPU."""
    if device.type != 'cuda' or not config.get('mixed_precision', True):
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_optimizer(crnn, iterations_per_epoch):
    """Return (optimizer, scheduler): AdamW with linear warmup followed by cosine decay, stepped per iteration."""
    optimizer = optim.AdamW(crnn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    warmup = config['warmup_iterations']
    total = config['epochs'] * iterations_per_epoch
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, milestones=[warmup], schedulers=[
        optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=config['lr_min']),
    ])
    return optimizer, scheduler


def save_checkpoint(crnn, optimizer, scheduler, scaler, iteration, epoch, evaluation):
    """Save the model weights as a plain state dict (loadable by every consumer) and the optimizer state separately."""
    weights_path = os.path.join(
        config['checkpoints_dir'],
        f'crnn_{iteration:06}_loss{evaluation["loss"]:.5f}_cer{evaluation["cer"]:.5f}_acc{evaluation["seq_acc"]:.5f}.pt')
    torch.save(crnn.state_dict(), weights_path)
    print('save model at ', weights_path)

    # The training state is only needed to resume; keep a single, latest copy to save space on the drive.
    train_state_path = os.path.join(config['checkpoints_dir'], 'crnn_train_state.pt')
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'iteration': iteration, 'epoch': epoch,
                'weights': os.path.basename(weights_path)}, train_state_path)
    print('save training state at ', train_state_path)


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    reload_train_state = config.get('reload_train_state')
    cache_dir = config.get('cache_dir')

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = select_amp_dtype(device)
    # All inputs have the same size, so let cuDNN pick the fastest kernels once.
    torch.backends.cudnn.benchmark = True
    print(f'device: {device}, mixed precision: {amp_dtype}')

    train_dataset = CssDataset(root_dir=data_dir, mode='train',
                                    img_height=img_height, img_width=img_width, cache_dir=cache_dir)
    valid_dataset = CssDataset(root_dir=data_dir, mode='dev',
                                    img_height=img_height, img_width=img_width, cache_dir=cache_dir)

    loader_options = {'num_workers': cpu_workers, 'collate_fn': css_collate_fn,
                      'pin_memory': device.type == 'cuda', 'persistent_workers': cpu_workers > 0}
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        **loader_options)
    # Keep the validation set in order so that evaluate() can attribute samples to their collections.
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_options)

    num_class = len(CssDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'],
                dropout=config['dropout'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device, weights_only=False))
    crnn.to(device)

    optimizer, scheduler = build_optimizer(crnn, len(train_loader))
    # The scaler is only active for float16; for bfloat16 and CPU it is a no-op pass-through.
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to(device)

    i = 1
    start_epoch = 1
    if reload_train_state:
        train_state = torch.load(reload_train_state, map_location=device, weights_only=False)
        optimizer.load_state_dict(train_state['optimizer'])
        scheduler.load_state_dict(train_state['scheduler'])
        scaler.load_state_dict(train_state['scaler'])
        i = train_state['iteration'] + 1
        start_epoch = train_state['epoch']  # the interrupted epoch is restarted (data is reshuffled anyway)
        print(f'resumed training state from {reload_train_state} (weights {train_state["weights"]}, iteration {i - 1})')

    assert save_interval % valid_interval == 0
    interval_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, scheduler, criterion, device, scaler, amp_dtype)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                elapsed = time.time() - interval_start
                print(f'train_batch_loss[ {i} ]:  {loss / train_size}  (lr {optimizer.param_groups[0]["lr"]:.2e}, '
                      f'{show_interval * train_size / elapsed:.0f} images/s)')
                interval_start = time.time()

            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: ' + format_evaluation(evaluation))

                if i % save_interval == 0:
                    save_checkpoint(crnn, optimizer, scheduler, scaler, i, epoch, evaluation)
                interval_start = time.time()

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()
