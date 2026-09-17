import os
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torchmetrics.text import CharErrorRate
from tqdm import tqdm

from dataset import CssDataset, css_collate_fn
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config

label2char = CssDataset.LABEL2CHAR

def labels_to_string(labels):
    """Convert a list of label indices to a string."""
    return ''.join(label2char[label] for label in labels)


def collection_of(path):
    """Return the collection an image belongs to"""
    return re.sub(r'(-aug|-\d+)', '', os.path.basename(os.path.dirname(path)))


def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    """Evaluate a model and return loss, CER and sequence accuracy (exact match), overall and per collection."""
    crnn.eval()

    tot_count = 0
    tot_loss = 0

    cer = CharErrorRate()
    exact_matches = 0
    per_collection = defaultdict(lambda: {'count': 0, 'exact_matches': 0, 'cer': CharErrorRate()})
    paths = getattr(dataloader.dataset, 'paths', None)
    sample_index = 0

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length

                pred_str = labels_to_string(pred)
                real_str = labels_to_string(real)

                cer.update(pred_str, real_str)
                exact_matches += pred_str == real_str

                if paths:
                    stats = per_collection[collection_of(paths[sample_index])]
                    stats['count'] += 1
                    stats['exact_matches'] += pred_str == real_str
                    stats['cer'].update(pred_str, real_str)
                sample_index += 1

            pbar.update(1)
        pbar.close()

    return {
        'loss': tot_loss / tot_count,
        'cer': cer.compute().item(),
        'seq_acc': exact_matches / tot_count,
        'collections': {
            name: {
                'count': stats['count'],
                'cer': stats['cer'].compute().item(),
                'seq_acc': stats['exact_matches'] / stats['count'],
            }
            for name, stats in sorted(per_collection.items())
        },
    }


def format_evaluation(evaluation):
    """Format an evaluation result as a one-line summary followed by one line per collection."""
    lines = ['loss={loss:.5f}, cer={cer:.5f}, seq_acc={seq_acc:.5f}'.format(**evaluation)]
    for name, stats in evaluation['collections'].items():
        lines.append(f'  {name}: n={stats["count"]}, cer={stats["cer"]:.5f}, seq_acc={stats["seq_acc"]:.5f}')
    return '\n'.join(lines)


def main():
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    test_dataset = CssDataset(root_dir=config['data_dir'], mode='test',
                                   img_height=img_height, img_width=img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=css_collate_fn)

    num_class = len(label2char) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device, weights_only=False))
    crnn.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: ' + format_evaluation(evaluation))


if __name__ == '__main__':
    main()
