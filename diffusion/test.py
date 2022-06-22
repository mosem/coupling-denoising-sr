import torch
import torchaudio

import data as Data
import model as Model
import argparse
import logging
from utils import setup_logger, parse, dict_to_nonedict, dict2str
import metrics as Metrics
from wandb_logger import WandbLogger
import os

from data.util import SequentialBinSampler, collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-disable_eval', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('--sr', type=int, default=16000)

    # parse configs
    args = parser.parse_args()
    opt = parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            sampler = SequentialBinSampler(val_set.get_file_lengths())
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase, sampler, collate_fn)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    source_sr = opt['datasets']['val']['source_sr']
    target_sr = opt['datasets']['val']['target_sr']

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    n_processed_batches = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    avg_pesq = 0.0
    avg_stoi = 0.0
    avg_sisnr = 0.0
    avg_lsd = 0.0
    avg_visqol = 0.0
    n_files_evaluated = 0

    for _, val_data in enumerate(val_loader):
        idx += 1
        filenames = val_data['filename']

        filenames_exist = [os.path.isfile(os.path.join(result_path, filename + '_target.wav')) for filename in filenames]
        if all(filenames_exist):
            logger.info(f'{idx}/{len(val_loader)}) all files already exists: {",".join(filenames)}')


            for filename in filenames:
                target_name = os.path.join(result_path, filename + '_target.wav')
                pred_name = os.path.join(result_path, filename + '_pr.wav')
                target_signal, _ = torchaudio.load(target_name)
                pred_signal, _ = torchaudio.load(pred_name)
                target_signal = Metrics.tensor2audio(target_signal)
                pred_signal = Metrics.tensor2audio(pred_signal)

                source_name = os.path.join(result_path, filename + '_source.wav')
                source_signal, _ = torchaudio.load(source_name)

                if not args.disable_eval:
                    pesq = Metrics.calculate_pesq(pred_signal.numpy(), target_signal.numpy(), target_sr)
                    stoi = Metrics.calculate_stoi(pred_signal.numpy(), target_signal.numpy(), target_sr)
                    sisnr = Metrics.calculate_sisnr(pred_signal, target_signal)
                    lsd = Metrics.calculate_lsd(pred_signal, target_signal)
                    visqol = Metrics.calculate_visqol(pred_signal.numpy(), target_signal.numpy(), filename, args.sr)
                    avg_pesq += pesq
                    avg_stoi += stoi
                    avg_sisnr += sisnr
                    avg_lsd += lsd
                    avg_visqol += visqol
                    n_files_evaluated += 1

                    if wandb_logger and opt['log_eval']:
                        wandb_logger.log_eval_data(filename, source_signal, pred_signal, target_signal,
                                                   target_sr,
                                                   pesq, stoi, sisnr, lsd, visqol)

                elif wandb_logger and opt['log_infer']:
                    wandb_logger.log_infer_data(filename, source_signal, pred_signal, target_signal,
                                                target_sr)

            continue

        logger.info(f'{idx}/{len(val_loader)}) Inferring {",".join(filenames)}.')
        n_processed_batches += 1

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        signals = diffusion.get_current_signals(need_source_raw=False)

        for i, filename in enumerate(filenames):
            target_signal = Metrics.tensor2audio(signals['target'][i])
            source_signal = Metrics.tensor2audio(signals['source'][i])
            pred_signal = signals['pred']

            signal_length = val_data['length'][i]

            target_signal = target_signal[:, :signal_length]
            source_signal = source_signal[:, :signal_length]
            pred_signal = pred_signal[:, :, :signal_length]

            sr_img_mode = 'grid'
            if sr_img_mode == 'grid':
                sample_num = pred_signal.shape[1]
                for sample_idx in range(0, sample_num - 1):
                    Metrics.save_audio(
                        Metrics.tensor2audio(pred_signal[i, sample_idx:sample_idx + 1, :]),
                        '{}/{}_pr_process_{}.wav'.format(result_path, filename, sample_idx), target_sr)

            Metrics.save_audio(
                Metrics.tensor2audio(pred_signal[i, -1:, :]),
                '{}/{}_pr.wav'.format(result_path, filename), target_sr)

            Metrics.save_audio(
                target_signal, '{}/{}_target.wav'.format(result_path, filename), target_sr)
            Metrics.save_audio(
                source_signal, '{}/{}_source.wav'.format(result_path, filename), target_sr)

            if not args.disable_eval:
                pesq = Metrics.calculate_pesq(pred_signal[i, -1:, :].numpy(), target_signal.numpy(), target_sr)
                stoi = Metrics.calculate_stoi(pred_signal[i, -1:, :].numpy(), target_signal.numpy(), target_sr)
                sisnr = Metrics.calculate_sisnr(pred_signal[i, -1:, :], target_signal)
                lsd = Metrics.calculate_lsd(pred_signal[i, -1:, :], target_signal)
                visqol = Metrics.calculate_visqol(pred_signal[i, -1:, :].numpy(), target_signal.numpy(), filename, args.sr)
                avg_pesq += pesq
                avg_stoi += stoi
                avg_sisnr += sisnr
                avg_lsd += lsd
                avg_visqol += visqol
                n_files_evaluated += 1

                if wandb_logger and opt['log_infer']:
                    wandb_logger.log_eval_data(filename, source_signal, Metrics.tensor2audio(pred_signal[i, -1:, :]), target_signal,
                                               target_sr,
                                               pesq, stoi, sisnr, lsd, visqol)
            elif wandb_logger and opt['log_infer']:
                wandb_logger.log_infer_data(filename, source_signal, Metrics.tensor2audio(pred_signal[i, -1:, :]), target_signal,
                                            target_sr)

    logger.info(f'Done. Processed {n_processed_batches}/{len(val_loader)} batches.')

    if not args.disable_eval:
        avg_pesq = avg_pesq / n_files_evaluated
        avg_stoi = avg_stoi / n_files_evaluated
        avg_sisnr = avg_sisnr / n_files_evaluated
        avg_lsd = avg_lsd / n_files_evaluated
        avg_visqol = avg_visqol / n_files_evaluated

        logger.info('# Validation # PESQ: {:.4}, STOI: {:.4},  SISNR: {:.4}, LSD: {:.4}, VISQOL: {:.4}'.format(
            avg_pesq, avg_stoi, avg_sisnr, avg_lsd, avg_visqol))

        if wandb_logger and opt['log_eval']:
            wandb_logger.log_metrics({
                'PESQ': avg_pesq,
                'STOI': avg_stoi,
                'SISNR': avg_sisnr,
                'LSD': avg_lsd,
                'VISQOL': avg_visqol
            })
            wandb_logger.log_eval_table(commit=True)

    elif wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)