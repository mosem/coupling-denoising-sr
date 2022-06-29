import torch
import torchaudio
import logging
import os
import hydra

try:
    import data as Data
    import model as Model
    import metrics as Metrics
    from utils import parse_dset_args
    from wandb_logger import WandbLogger
except ModuleNotFoundError:
    import sys

    sys.path.append(sys.path[0] + '/..')
    import data as Data
    import model as Model
    import metrics as Metrics
    from utils import parse_dset_args
    from wandb_logger import WandbLogger


def test(args, logger, wandb_logger=None):

    # dataset
    val_set = Data.create_dataset(args.dset, 'val')
    val_loader = Data.create_dataloader(val_set, args.dset, 'val')
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(args)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        args.noise_schedule.val, schedule_phase='val')

    target_sr = args.dset.target_sr

    logger.info('Begin Model Inference.')
    idx = 0
    n_processed_batches = 0

    result_path = args.path.results

    avg_pesq = 0.0
    avg_stoi = 0.0
    avg_sisnr = 0.0
    avg_lsd = 0.0
    avg_visqol = 0.0
    n_files_evaluated = 0

    for _, val_data in enumerate(val_loader):
        idx += 1
        filenames = val_data['filename']

        filenames_exist = [os.path.isfile(os.path.join(result_path, filename + '_target.wav')) for filename in
                           filenames]
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

                metrics = None
                if not args.disable_eval:
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
                    n_files_evaluated += 1

                    metrics = {'pesq': pesq, 'stoi':stoi, 'sisnr': sisnr, 'lsd': lsd, 'visqol': visqol}

                if wandb_logger:
                    wandb_logger.log_data(filename, source_signal, pred_signal, target_signal,
                                                target_sr, metrics)

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

            metrics = None
            if not args.disable_eval:
                pesq = Metrics.calculate_pesq(pred_signal[i, -1:, :].numpy(), target_signal.numpy(), target_sr)
                stoi = Metrics.calculate_stoi(pred_signal[i, -1:, :].numpy(), target_signal.numpy(), target_sr)
                sisnr = Metrics.calculate_sisnr(pred_signal[i, -1:, :], target_signal)
                lsd = Metrics.calculate_lsd(pred_signal[i, -1:, :], target_signal)
                visqol = Metrics.calculate_visqol(pred_signal[i, -1:, :].numpy(), target_signal.numpy(), filename,
                                                  target_sr)
                avg_pesq += pesq
                avg_stoi += stoi
                avg_sisnr += sisnr
                avg_lsd += lsd
                avg_visqol += visqol
                n_files_evaluated += 1

                metrics = {'pesq': pesq, 'stoi': stoi, 'sisnr': sisnr, 'lsd': lsd, 'visqol': visqol}

            if wandb_logger:
                wandb_logger.log_data(filename, source_signal, Metrics.tensor2audio(pred_signal[i, -1:, :]),
                                            target_signal,
                                            target_sr, metrics)

    logger.info(f'Done. Processed {n_processed_batches}/{len(val_loader)} batches.')

    if not args.disable_eval:
        avg_pesq = avg_pesq / n_files_evaluated
        avg_stoi = avg_stoi / n_files_evaluated
        avg_sisnr = avg_sisnr / n_files_evaluated
        avg_lsd = avg_lsd / n_files_evaluated
        avg_visqol = avg_visqol / n_files_evaluated

        logger.info('# Validation # PESQ: {:.4}, STOI: {:.4},  SISNR: {:.4}, LSD: {:.4}, VISQOL: {:.4}'.format(
            avg_pesq, avg_stoi, avg_sisnr, avg_lsd, avg_visqol))

        if wandb_logger and args.wandb.log_eval:
            wandb_logger.log_metrics(metrics)
            wandb_logger.log_metrics_table({
                'pesq': avg_pesq,
                'stoi': avg_stoi,
                'sisnr': avg_sisnr,
                'lsd': avg_lsd,
                'visqol': avg_visqol
            })


    if wandb_logger:
        wandb_logger.log_results_table(commit=True)

def _main(args):
    args['phase'] = 'test'
    args['resume'] = True
    if not args.resume_state:
        # get last state
        sorted_checkpoint_files = sorted(os.listdir(args.path.checkpoint))
        last_state = '_'.join(sorted_checkpoint_files[-1].split('_')[:2])
        args.resume_state = last_state

    parse_dset_args(args.dset)

    os.makedirs(args.path.results, exist_ok=True)

    logger = logging.getLogger('base')
    logger.info("For logs and samples check %s", os.getcwd())
    logger.info(args)

    # Initialize WandbLogger
    if args.wandb.enable:
        wandb_logger = WandbLogger(args)
    else:
        wandb_logger = None

    test(args, logger, wandb_logger)


@hydra.main(config_path="conf", config_name="main_config")
def main(args):
    try:
        _main(args)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("Some error happened")
        os._exit(1)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main()

