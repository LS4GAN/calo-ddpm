import argparse

import os
from jetgen import ROOT_OUTDIR, train

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train centrality 4 DDPM'
    )

    parser.add_argument(
        '--batch-size',
        default = 128,
        dest    = 'batch_size',
        help    = 'train batch size',
        type    = int
    )

    parser.add_argument(
        '--epochs',
        default = 500,
        dest    = 'epochs',
        help    = 'number of training epochs',
        type    = int
    )

    parser.add_argument(
        '--seed',
        default = 0,
        dest    = 'seed',
        help    = 'random seed',
        type    = int
    )

    parser.add_argument(
        '-T', '--diffusion-steps',
        default = 8000,
        dest    = 'T',
        help    = 'number of diffusion steps',
        type    = int
    )

    parser.add_argument(
        '--betaT',
        default = 0.10,
        dest    = 'beta_t',
        help    = r'value of $\beta_T$',
        type    = float,
    )

    parser.add_argument(
        '--lr',
        default = 1e-4,
        dest    = 'lr',
        help    = 'learning rate',
        type    = float,
    )

    return parser.parse_args()

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'noise',
                    'shape'  : (1, 24, 64),
                    'length' : 1000000,
                    'mu'     : 0,
                    'sigma'  : 1,
                },
                'shape'           : (1, 24, 64),
                'transform_train' : None,
                'transform_test'  : None,
            },
            {
                'dataset'         : {
                    'name'   : 'h5array-domain-hierarchy',
                    'path'   : 'sphenix/cent4',
                    'domain' : 'b',
                },
                'shape'           : (1, 24, 64),
                'transform_train' : 'to-tensor',
                'transform_test'  : 'to-tensor',
            },
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'        : cmdargs.epochs,
    'discriminator' : None,
    'generator'     : {
        'model' : 'iddpm-unet',
        'model_args' : {
            'model_channels'        : 32,
            'num_res_blocks'        : 2,
            'attention_resolutions' : (),
            'dropout'               : 0,
            'channel_mult'          : (1, 2, 4),
            'num_classes'           : None,
            'use_checkpoint'        : False,
            'num_heads'             : 1,
            'num_heads_upsample'    : -1,
            'use_scale_shift_norm'  : True,
        },
        'optimizer' : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr,
        },
    },
    'model'      : 'ddpm',
    'model_args' : {
        'avg_momentum' : 0.9999,
        'data_norm' : {
            'name'     : 'log',
            'clip_min' : 1e-3,
        },
        'seed'   : 0,
        'vsched' : {
            'name'  : 'linear',
            'T'     : cmdargs.T,
            'beta1' : 1    / cmdargs.T * cmdargs.beta_t,
            'betaT' : 1000 / cmdargs.T * cmdargs.beta_t,
        },
    },
    'seed'             : cmdargs.seed,
    'scheduler'        : None,
    'loss'             : 'l2',
    'steps_per_epoch'  : 2000,
    'transfer'         : None,
    'gradient_penalty' : None,
# args
    'label'  : (
        f'batch_size({cmdargs.batch_size})_epochs({cmdargs.epochs})'
        f'_lr({cmdargs.lr})_seed({cmdargs.seed})'
        f'_T({cmdargs.T})_betaT({cmdargs.beta_t})'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'calo-ddpm', 'sphenix', 'cent4_ddpm'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
}

train(args_dict)

