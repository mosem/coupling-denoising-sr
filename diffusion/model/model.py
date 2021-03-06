import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import diffusion.model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, args):
        super(DDPM, self).__init__(args)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(args))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            args.noise_schedule.train, schedule_phase='train')
        if self.args.phase == 'train':
            self.netG.train()
            # find the parameters to optimize
            if args.model.finetune_norm:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=args.train.optimizer.lr)
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        loss = self.netG(self.data)
        # need to average in multi-gpu
        b, c, t = self.data['target'].shape
        loss = loss.sum()/int(b*c*t)
        loss.backward()
        self.optG.step()

        # set log
        self.log_dict['loss'] = loss.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.pred = self.netG.module.super_resolution(
                    self.data['source'], continous)
            else:
                new_pred = self.netG.super_resolution(
                    self.data['source'], continous)
                self.pred = new_pred
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_config, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_config, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_config, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_signals(self, need_source_raw=False):
        out_dict = OrderedDict()

        out_dict['pred'] = self.pred.detach().float().cpu()
        out_dict['source'] = self.data['source'].detach().float().cpu()
        out_dict['target'] = self.data['target'].detach().float().cpu()
        if need_source_raw and 'source_raw' in self.data:
            out_dict['source_raw'] = self.data['source_raw'].detach().float().cpu()
        else:
            out_dict['source_raw'] = out_dict['source']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.args.path.checkpoint, 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.args.path.checkpoint, 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        if self.args.resume and self.args.resume_state:
            load_path = os.path.join(self.args.path.checkpoint, self.args.resume_state)
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.args.model.finetune_norm))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.args.phase == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
