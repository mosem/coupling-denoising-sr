import logging
logger = logging.getLogger('base')


def create_model(args):
    from .model import DDPM as M
    m = M(args)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
