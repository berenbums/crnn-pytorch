from collections import defaultdict

import numpy as np
from scipy.special import logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01


def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels


def beam_search_nbest(emission_log_prob, blank=0, beam_size=10, emission_threshold=None, n_best=None):
    """Beam search over CTC paths returning the n most probable label sequences with their probabilities.

    Same search as the original beam_search_decode (prune paths per timestep, merge paths that collapse to the
    same labels at the end), vectorised over the beam x class matrix. Probabilities are normalised over the merged
    sequences that survived the search, so the top entry's probability is a usable confidence."""
    if emission_threshold is None:
        emission_threshold = np.log(DEFAULT_EMISSION_THRESHOLD)
    length, class_count = emission_log_prob.shape
    emission_log_prob = np.where(emission_log_prob < emission_threshold, NINF, emission_log_prob)

    prefixes = [()]
    scores = np.zeros(1)
    for t in range(length):
        # (beam, class): score of every one-step extension of every beam.
        candidates = scores[:, None] + emission_log_prob[t][None, :]
        flat = candidates.ravel()
        if flat.size > beam_size:
            top = np.argpartition(-flat, beam_size - 1)[:beam_size]
        else:
            top = np.arange(flat.size)
        top = top[flat[top] > NINF]
        if top.size == 0:
            # Every class fell below the threshold at this timestep: keep the beams and treat the step as blank.
            prefixes = [prefix + (blank,) for prefix in prefixes]
            continue
        scores = flat[top]
        prefixes = [prefixes[index // class_count] + (index % class_count,) for index in top]

    # Merge paths that collapse to the same label sequence.
    merged = {}
    for prefix, score in zip(prefixes, scores):
        labels = tuple(_reconstruct(prefix, blank))
        merged[labels] = logsumexp([merged.get(labels, NINF), score])
    labels_list = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:n_best]
    total = logsumexp(list(merged.values()))
    return [(list(labels), float(np.exp(score - total))) for labels, score in labels_list]


def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))
    return beam_search_nbest(emission_log_prob, blank, beam_size, emission_threshold, n_best=1)[0][0]


def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [(tuple(), (0, NINF))]  # (prefix, (blank_log_prob, non_blank_log_prob))
    # initial of beams: (empty_str, (log(1.0), log(0.0)))

    for t in range(length):
        new_beams_dict = defaultdict(lambda: (NINF, NINF))  # log(0.0) = NINF

        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                # if new_prefix == prefix
                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                # if new_prefix == prefix + (c,)
                new_prefix = prefix + (c,)
                new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                if c != end_t:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )
                else:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob])
                    )

        # sorted by log(blank_prob + non_blank_prob)
        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    labels = list(beams[0][0])
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10, n_best=None):
    """Decode a (length, batch, class) log-probability tensor.

    Returns one label list per image, or with n_best (beam_search only) a list of (labels, probability) pairs per
    image, most probable first."""
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]
    to_chars = (lambda labels: [label2char[l] for l in labels]) if label2char else (lambda labels: labels)

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        if n_best:
            hypotheses = beam_search_nbest(emission_log_prob, blank=blank, beam_size=beam_size, n_best=n_best)
            decoded_list.append([(to_chars(labels), probability) for labels, probability in hypotheses])
        else:
            decoded_list.append(to_chars(decoder(emission_log_prob, blank=blank, beam_size=beam_size)))
    return decoded_list
