import torch
import data as Data
import model as Model
import argparse
import logging
from utils import setup_logger, parse, parse_dset_args, dict_to_nonedict, dict2str
import metrics as Metrics
from wandb_logger import WandbLogger
import os
import hydra


def train(args, logger, wandb_logger=None):
    val_step=0

    # dataset
    train_set = Data.create_dataset(args.dset, 'train')
    train_loader = Data.create_dataloader(train_set, args.dset, 'train')

    val_set = Data.create_dataset(args.dset, 'val')
    val_loader = Data.create_dataloader(val_set, args.dset, 'val')

    logger.info('Initial Dataset Finished')

    logger.info(f'train set length: {len(train_set)}')
    logger.info(f'train loader length: {len(train_loader)}')

    logger.info(f'val set length: {len(val_set)}')
    logger.info(f'val loader length: {len(val_loader)}')

    source_sr = args.dset.source_sr
    target_sr = args.dset.target_sr

    # model
    diffusion = Model.create_model(args)
    logger.info('Initial Model Finished')
    n_params = sum(p.numel() for p in diffusion.netG.parameters() if p.requires_grad)
    mb = n_params * 4 / 2 ** 20
    logger.info(f"{args.model.hdemucs}: parameters: {n_params}, size: {mb} MB")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = args.train.n_iter

    if args.path.resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        args.beta_schedule.train, schedule_phase='train')

    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if current_step % args.train.print_freq == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4} '.format(k, float(v))
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics(logs)

            # validation
            if current_step % args == 0:
                avg_pesq = 0.0
                avg_stoi = 0.0
                avg_sisnr = 0.0
                avg_lsd = 0.0
                avg_visqol = 0.0
                idx = 0
                result_path = '{}/{}'.format(args.path.results, current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    args.model.beta_scheulde.val, schedule_phase='val')
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    signals = diffusion.get_current_signals()

                    pred_signal = Metrics.tensor2audio(signals['pred'])[0, -1:, :]
                    target_signal = Metrics.tensor2audio(signals['target'])
                    source_signal = Metrics.tensor2audio(signals['source'])

                    filename = val_data['filename'][0]

                    # generation
                    Metrics.save_audio(
                        target_signal, '{}/{}_target.wav'.format(result_path, filename), target_sr)
                    Metrics.save_audio(
                        pred_signal, '{}/{}_pr.wav'.format(result_path, filename), target_sr)
                    Metrics.save_audio(
                        source_signal, '{}/{}_source.wav'.format(result_path, filename), source_sr)

                    pesq = Metrics.calculate_pesq(pred_signal.numpy(), target_signal.numpy(), target_sr)
                    stoi = Metrics.calculate_stoi(pred_signal.numpy(), target_signal.numpy(), target_sr)
                    sisnr = Metrics.calculate_sisnr(pred_signal, target_signal)
                    lsd = Metrics.calculate_lsd(pred_signal, target_signal)
                    visqol = Metrics.calculate_visqol(pred_signal.numpy(), target_signal.numpy(), filename, target_sr)
                    avg_pesq += pesq
                    avg_stoi += stoi
                    avg_sisnr += sisnr
                    avg_lsd += lsd
                    avg_visqol += visqol

                    if wandb_logger:
                        wandb_logger.log_audio(filename, pred_signal, target_signal, source_signal, pesq, stoi, sisnr,
                                               lsd, visqol,
                                               current_step, target_sr)

                avg_pesq = avg_pesq / idx
                avg_stoi = avg_stoi / idx
                avg_sisnr = avg_sisnr / idx
                avg_lsd = avg_lsd / idx
                avg_visqol = avg_visqol / idx
                diffusion.set_new_noise_schedule(
                    args.beta_schedule.train, schedule_phase='train')
                # log
                logger.info('# Validation # PESQ: {:.4}, STOI: {:.4},  SISNR: {:.4}, LSD: {:.4}, VISQOL: {:.4}'.format(
                    avg_pesq, avg_stoi, avg_sisnr, avg_lsd, avg_visqol))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info(
                    '<epoch:{:3d}, iter:{:8,d}> pesq: {:.4}, stoi: {:.4}, sisnr: {:.4}, lsd: {:.4}, visqol: {:.4}'.format(
                        current_epoch, current_step, avg_pesq, avg_stoi, avg_sisnr, avg_lsd, avg_visqol))

                if wandb_logger:
                    val_step += 1
                    wandb_logger.log_metrics({
                        'validation/val_pesq': avg_pesq,
                        'validation/val_stoi': avg_stoi,
                        'validation/val_sisnr': avg_sisnr,
                        'validation/val_lsd': avg_lsd,
                        'validation/val_visqol': avg_visqol,
                        'validation/val_step': val_step
                    })

            if current_step % args.train.save_checkpoint_freq == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

                if wandb_logger and args.wandb.log_wandb_ckpt:
                    wandb_logger.log_checkpoint(current_epoch, current_step)

        if wandb_logger:
            wandb_logger.log_metrics({'epoch': current_epoch - 1})

    # save model
    logger.info('End of training.')


def _main(args):
    print(args)
    parse_dset_args(args.dset)

    setup_logger(None, args.path.log,
                 'train', level=logging.INFO, screen=True)
    setup_logger('val', args.path.log, 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(dict2str(args))

    # Initialize WandbLogger
    if args.wandb.enable:
        import wandb

        wandb_logger = WandbLogger(args)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")

    else:
        wandb_logger = None

    train(args, logger, wandb_logger)


@hydra.main(config_path="config", config_name="main_config")
def main(args):
    try:
        _main(args)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
    #                     help='JSON file for configuration')
    # parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
    #                     help='Run either train(training) or val(generation)', default='train')
    # parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    # parser.add_argument('-debug', '-d', action='store_true')
    # parser.add_argument('-enable_wandb', action='store_true')
    # parser.add_argument('-log_wandb_ckpt', action='store_true')
    # parser.add_argument('-log_eval', action='store_true')
    #
    # # parse configs
    # args = parser.parse_args()
    # opt = parse(args)
    # Convert to NoneDict, which return None for missing key.
    # opt = dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()





