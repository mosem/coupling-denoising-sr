import argparse
import os
import logging

import torchaudio

import metrics as Metrics
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args()

    logger = logging.getLogger('base')

    target_names = list(glob.glob('{}/*_target.wav'.format(args.path)))
    pred_names = list(glob.glob('{}/*_pr.wav'.format(args.path)))

    target_names.sort()
    pred_names.sort()

    avg_pesq = 0.0
    avg_stoi = 0.0
    avg_sisnr = 0.0
    avg_lsd = 0.0
    avg_visqol = 0.0
    idx = 0
    for target_name, pred_name in zip(target_names, pred_names):
        idx += 1
        target_idx = target_name.rsplit("_target")[0]
        pred_idx = pred_name.rsplit("_pr")[0]
        assert target_idx == pred_idx, 'Image target_idx:{target_idx}!=pred_idx:{pred_idx}'.format(
            target_idx, pred_idx)

        target_signal, _ = torchaudio.load(target_name)
        pred_signal, _ = torchaudio.load(pred_name)
        target_signal = Metrics.tensor2audio(target_signal)
        pred_signal = Metrics.tensor2audio(pred_signal)
        pesq = Metrics.calculate_pesq(pred_signal.numpy(), target_signal.numpy(), args.sr)
        stoi = Metrics.calculate_stoi(pred_signal.numpy(), target_signal.numpy(), args.sr)
        sisnr = Metrics.calculate_sisnr(pred_signal, target_signal)
        lsd = Metrics.calculate_lsd(pred_signal, target_signal)
        visqol = Metrics.calculate_visqol(pred_signal.numpy(), target_signal.numpy(), os.path.basename(target_idx), args.sr)
        avg_pesq += pesq
        avg_stoi += stoi
        avg_sisnr += sisnr
        avg_lsd += lsd
        avg_visqol += visqol

        if idx % 20 == 0:
            logger.info('Audio:{}, PESQ:{:.4f}, STOI:{:.4f}, SISNR:{:.4f}, LSD:{:.4f}, VISQOL: {:.4f}'
                                                            .format(idx, pesq, stoi, sisnr, lsd, visqol))

    avg_pesq = avg_pesq / idx
    avg_stoi = avg_stoi / idx
    avg_sisnr = avg_sisnr / idx
    avg_lsd = avg_lsd / idx
    avg_visqol = avg_visqol / idx

    # log
    logger.info('# Validation # PESQ: {:.4}, STOI: {:.4},  SISNR: {:.4}, LSD: {:.4}, VISQOL: {:.4}'.format(
        avg_pesq, avg_stoi, avg_sisnr, avg_lsd, avg_visqol))
