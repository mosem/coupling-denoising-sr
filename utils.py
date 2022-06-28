import hydra


def parse_dset_args(dset_args):
    dset_args['train_json_dir'] = hydra.utils.to_absolute_path(dset_args.train_json_dir)
    dset_args['val_json_dir'] = hydra.utils.to_absolute_path(dset_args.val_json_dir)
