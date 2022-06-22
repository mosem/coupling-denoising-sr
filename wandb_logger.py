import os
from torchaudio.transforms import Spectrogram
import numpy as np
import cv2

def scale_minmax(X, min=0.0, max=1.0):
    isnan = np.isnan(X).any()
    isinf = np.isinf(X).any()
    if isinf:
        X[X == np.inf] = 1e9
        X[X == -np.inf] = 1e-9
    if isnan:
        X[X == np.nan] = 1e-9
    # logger.info(f'isnan: {isnan}, isinf: {isinf}, max: {X.max()}, min: {X.min()}')

    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def convert_spectrogram_to_heatmap(spectrogram):
    spectrogram += 1e-9
    spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8).squeeze()
    spectrogram = np.flip(spectrogram, axis=0)
    spectrogram = 255 - spectrogram
    # spectrogram = (255 * (spectrogram - np.min(spectrogram)) / np.ptp(spectrogram)).astype(np.uint8).squeeze()[::-1,:]
    heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


SPECTOGRAM_EPSILON = 1e-13

class WandbLogger:
    """
    Log using `Weights and Biases`.
    """
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        WANDB_ENTITY = 'huji-dl-audio-lab'

        self._wandb = wandb
        self.spectrogram_transform = Spectrogram(n_fft=1024, win_length=240, hop_length=50)

        self.files_initially_logged = set()

        # Initialize a W&B run
        if self._wandb.run is None:
            if 'id' in opt['wandb']:
                self._wandb.init(
                    id=opt['wandb']['id'],
                    resume='must',
                    project=opt['wandb']['project'],
                    entity=WANDB_ENTITY,
                    config=opt,
                    group=opt['wandb']['group'],
                    name=opt['wandb']['name'],
                    dir='./experiments',
                )
            else:
                self._wandb.init(
                    project=opt['wandb']['project'],
                    entity=WANDB_ENTITY,
                    config=opt,
                    group=opt['wandb']['group'],
                    name=opt['wandb']['name'],
                    dir='./experiments'
                )

        self.config = self._wandb.config

        self.disable_eval = self.config.get('disable_eval', None)

        if self.config.get('log_eval', None):
            self.eval_table = self._wandb.Table(columns=['source_audio',
                                                         'pred_audio',
                                                         'target_audio',
                                                         'source_spec',
                                                         'pred_spec',
                                                         'target_spec',
                                                         'pesq',
                                                         'stoi',
                                                         'sisnr',
                                                         'lsd',
                                                         'visqol'])
        else:
            self.eval_table = None

        if self.config.get('log_infer', None):
            self.infer_table = self._wandb.Table(columns=['source_audio',
                                                          'pred_audio',
                                                          'target_audio',
                                                          'source_spec',
                                                          'pred_spec',
                                                          'target_spec'])
        else:
            self.infer_table = None

    def log_metrics(self, metrics, commit=True): 
        """
        Log train/validation metrics onto W&B.

        metrics: dictionary of metrics to be logged
        """
        self._wandb.log(metrics, commit=commit)

    def log_audio(self, filename, pred_signal, target_signal, source_signal, pesq, stoi, sisnr, lsd, visqol, epoch, sr):
        pred_wandb_data = self.convert_signal_to_wandb_data(pred_signal, sr, f'{filename}_pred')

        wandb_dict = {f'test samples/{filename}/pesq': pesq,
                      f'test samples/{filename}/stoi': stoi,
                      f'test samples/{filename}/lsd': lsd,
                      f'test samples/{filename}/sisnr': sisnr,
                      f'test samples/{filename}/visqol': visqol,
                      f'test samples/{filename}/spectrogram': pred_wandb_data['spec'],
                      f'test samples/{filename}/audio': pred_wandb_data['audio']}

        if filename not in self.files_initially_logged:
            self.files_initially_logged.add(filename)
            target_name = f'{filename}_target'
            target_wandb_data = self.convert_signal_to_wandb_data(target_signal, sr, target_name)

            wandb_dict.update({f'test samples/{filename}/{target_name}_spectrogram': target_wandb_data['spec'],
                               f'test samples/{filename}/{target_name}_audio': target_wandb_data['audio']})

            source_name = f'{filename}_source'
            source_wandb_data = self.convert_signal_to_wandb_data(source_signal, sr, source_name)
            wandb_dict.update({f'test samples/{filename}/{source_name}_spectrogram': source_wandb_data['spec'],
                               f'test samples/{filename}/{source_name}_audio': source_wandb_data['audio']})

        self._wandb.log(wandb_dict,
                  step=epoch)

    def log_checkpoint(self, current_epoch, current_step):
        """
        Log the model checkpoint as W&B artifacts

        current_epoch: the current epoch 
        current_step: the current batch step
        """
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        gen_path = os.path.join(
            self.config.path['checkpoint'], 'I{}_E{}_gen.pth'.format(current_step, current_epoch))
        opt_path = os.path.join(
            self.config.path['checkpoint'], 'I{}_E{}_opt.pth'.format(current_step, current_epoch))

        model_artifact.add_file(gen_path)
        model_artifact.add_file(opt_path)
        self._wandb.log_artifact(model_artifact, aliases=["latest"])


    def convert_signal_to_wandb_data(self, signal, sr, caption, add_eps=False):


        spectrogram = self.spectrogram_transform(signal)
        if add_eps:
            spectrogram += SPECTOGRAM_EPSILON
        spectrogram = spectrogram.log2()[0, :, :].numpy()

        wandb_audio = self._wandb.Audio(signal.squeeze().numpy(), sample_rate=sr,
                                        caption=caption)
        wandb_spectrogram = self._wandb.Image(convert_spectrogram_to_heatmap(spectrogram))

        return {'audio': wandb_audio, 'spec': wandb_spectrogram}



    def log_eval_data(self, filename, source_signal, pred_signal, target_signal, target_sr,
                                                                    pesq, stoi, sisnr, lsd, visqol):
        """
        Add data row-wise to the initialized table.
        """

        source_wandb_data = self.convert_signal_to_wandb_data(source_signal, target_sr,
                                                              filename + '_source', add_eps=True)
        pred_wandb_data = self.convert_signal_to_wandb_data(pred_signal, target_sr,
                                                             filename + '_pr', add_eps=True)
        target_wandb_data = self.convert_signal_to_wandb_data(target_signal, target_sr,
                                                              filename + '_target', add_eps=True)


        self.eval_table.add_data(
            source_wandb_data['audio'],
            pred_wandb_data['audio'],
            target_wandb_data['audio'],
            source_wandb_data['spec'],
            pred_wandb_data['spec'],
            target_wandb_data['spec'],
            pesq,
            stoi,
            sisnr,
            lsd,
            visqol
        )

    def log_infer_data(self, filename, source_signal, pred_signal, target_signal, target_sr):
        """
                Add data row-wise to the initialized table.
                """

        source_wandb_data = self.convert_signal_to_wandb_data(source_signal, target_sr,
                                                              filename + '_source', add_eps=True)
        pred_wandb_data = self.convert_signal_to_wandb_data(pred_signal, target_sr,
                                                            filename + '_pr', add_eps=True)
        target_wandb_data = self.convert_signal_to_wandb_data(target_signal, target_sr,
                                                              filename + '_target', add_eps=True)

        self.infer_table.add_data(
            source_wandb_data['audio'],
            pred_wandb_data['audio'],
            target_wandb_data['audio'],
            source_wandb_data['spec'],
            pred_wandb_data['spec'],
            target_wandb_data['spec']
        )

    def log_eval_table(self, commit=False):
        """
        Log the table
        """
        if self.eval_table and not self.disable_eval:
            self._wandb.log({'eval_data': self.eval_table}, commit=commit)
        if self.infer_table:
            self._wandb.log({'infer_data': self.infer_table}, commit=commit)
