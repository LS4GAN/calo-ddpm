import argparse

import os
from jetgen import ROOT_OUTDIR, train

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train centrality 0 DCGAN'
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
        '--loss',
        default = 'lsgan',
        choices = [ 'lsgan', 'wgan' ],
        dest    = 'loss',
        help    = 'GAN loss',
        type    = str
    )

    parser.add_argument(
        '--gp-center',
        default = 1,
        dest    = 'gp_center',
        help    = 'center value of gradient penalty',
        type    = float,
    )

    parser.add_argument(
        '--gp-magnitude',
        default = 10,
        dest    = 'gp_magnitude',
        help    = 'magnitude value of gradient penalty',
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
                    'shape'  : (100,),
                    'length' : 1000000,
                    'mu'     : 0,
                    'sigma'  : 1,
                },
                'shape'           : (100,),
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
    'epochs'      : cmdargs.epochs,
    'discriminator' : {
        'model'      : 'dcgan',
        'model_args' : {
            'features_list' : [ 64, 128, 256, 512, 1024 ],
            'activ' : {
                'name' : 'leakyrelu',
                'negative_slope' : 0.2,
            },
            'norm'  : 'batch',
        },
        'optimizer'  :{
            'name'  : 'Adam',
            'lr'    : cmdargs.lr,
            'betas' : (0.5, 0.9),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : False,
    },
    'generator' : {
        'model' : 'dcgan',
        'model_args' : {
            'features_list' : [ 1024, 512, 256, 128, 64 ],
            'norm'         : 'batch',
            'activ'        : 'relu',
            'activ_output' : None,
        },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr,
            'betas' : (0.5, 0.9),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'gan',
    'model_args' : {
        'avg_momentum'    : 0,
        'head_queue_size' : 0,
        'data_norm'       : {
            'name'     : 'log',
            'clip_min' : 1e-3,
        },
        'head_config' : 'idt',
    },
    'seed'             : cmdargs.seed,
    'scheduler'        : None,
    'loss'             : cmdargs.loss,
    'steps_per_epoch'  : 2000,
    'transfer'         : None,
    'gradient_penalty' : {
        'center'    : cmdargs.gp_center,
        'lambda_gp' : cmdargs.gp_magnitude,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    } if cmdargs.gp_magnitude else None,
# args
    'label'  : (
        f'batch_size({cmdargs.batch_size})_epochs({cmdargs.epochs})'
        f'_lr({cmdargs.lr})_seed({cmdargs.seed})'
        f'_gp({cmdargs.gp_magnitude}-{cmdargs.gp_center})'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'calo-ddpm', 'sphenix', 'cent4_gan'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
}

train(args_dict)

