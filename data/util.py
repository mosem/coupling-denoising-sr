import torch


def collate_fn(data):
    '''
    We should build a custom collate_fn rather than using default collate_fn,
    as the size of every sentence is different and merging sequences (including padding)
    is not supported in default.
    Args:
        data: list of dicts {'HR': hr_sig, 'SR': sr_sig, 'filename': filename}
    Return:
        dictionary of batches: {'HR': hr_sig_batch, 'SR': sr_sig_batch, 'filename': hr_filename_batch}
    '''
    lengths = [d['HR'].size(-1) for d in data]
    sig_channels = data[0]['SR'].size(0)
    max_len = max(lengths)
    hr_padded = torch.zeros(len(data), sig_channels, max_len).type(data[0]['HR'].type())
    sr_padded = torch.zeros(len(data), sig_channels, max_len).type(data[0]['SR'].type())

    for i in range(len(data)):
        sig_len = data[i]['HR'].size(-1)
        hr_padded[i,:,:sig_len] = data[i]['HR']
        sr_padded[i,:,:sig_len] = data[i]['SR']

    filenames = [d['filename'] for d in data]
    file_lengths = [d['length'] for d in data]


    return {'HR': hr_padded, 'SR': sr_padded, 'filename': filenames, 'length': file_lengths}


class SequentialBinSampler(torch.utils.data.Sampler):
    def __init__(self, file_lengths):
        self.file_lengths = file_lengths
        self.idx_len_pairs = [(i,length) for i,length in enumerate(self.file_lengths)]
        self.indices_sorted_by_len = [x[0] for x in sorted(self.idx_len_pairs, key=lambda x: x[1])]

    def __len__(self):
        return len(self.file_lengths)

    def __iter__(self):
        return iter(self.indices_sorted_by_len)
