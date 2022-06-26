'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_config, phase, sampler=None, collate_fn=None):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_config.train_batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=dataset_config.train_num_workers,
            pin_memory=True)
    elif phase == 'val':
        batch_size = dataset_config.val_batch_size if 'val_batch_size' in dataset_config else 1
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=dataset_config.val_num_workers, pin_memory=True,
            sampler=sampler,
            collate_fn=collate_fn)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_config, phase):
    '''create dataset'''
    json_dir = dataset_config.train_json_dir if phase == 'train' else dataset_config.val_json_dir
    segment = dataset_config.train_segment  if phase == 'train' else None
    stride = dataset_config.train_stride if phase == 'train' else None
    pad_to_output_length = True if phase == 'val' else False
    data_len = dataset_config.train_data_len if phase == 'train' else dataset_config.val_data_len

    # from data.LRHR_dataset import LRHRDataset as D
    from data.audio_dataset import AudioDataset as D
    dataset = D(json_dir=json_dir,
                source_sr=dataset_config.source_sr,
                target_sr=dataset_config.target_sr,
                segment=segment,
                stride=stride,
                pad_to_output_length=pad_to_output_length,
                data_len=data_len
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_config.name))
    return dataset
