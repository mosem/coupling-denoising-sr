import torch


def collate_fn(data):
    '''
    We should build a custom collate_fn rather than using default collate_fn,
    as the size of every sentence is different and merging sequences (including padding)
    is not supported in default.
    Args:
        data: list of dicts {'target': target_signal, 'source': source_signal, 'filename': filename, 'length': target/source signal_length}
    Return:
        dictionary of batches: {'target': target_signal_batch, 'source': source_signal_batch, 'filename': filename_batch, 'length', length_batch}
    '''
    lengths = [d['target'].size(-1) for d in data]
    sig_channels = data[0]['source'].size(0)
    max_len = max(lengths)
    target_padded = torch.zeros(len(data), sig_channels, max_len).type(data[0]['target'].type())
    source_padded = torch.zeros(len(data), sig_channels, max_len).type(data[0]['source'].type())

    for i in range(len(data)):
        target_length = data[i]['target'].size(-1)
        source_length = data[i]['source'].size(-1)
        assert target_length == source_length
        target_padded[i,:,:target_length] = data[i]['target']
        source_padded[i,:,:source_length] = data[i]['source']

    filenames = [d['filename'] for d in data]
    file_lengths = [d['length'] for d in data]


    return {'target': target_padded, 'source': source_padded, 'filename': filenames, 'length': file_lengths}


class SequentialBinSampler(torch.utils.data.Sampler):
    def __init__(self, file_lengths):
        self.file_lengths = file_lengths
        self.idx_len_pairs = [(i,length) for i,length in enumerate(self.file_lengths)]
        self.indices_sorted_by_len = [x[0] for x in sorted(self.idx_len_pairs, key=lambda x: x[1])]

    def __len__(self):
        return len(self.file_lengths)

    def __iter__(self):
        return iter(self.indices_sorted_by_len)
