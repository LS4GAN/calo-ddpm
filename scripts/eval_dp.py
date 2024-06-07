#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np

from jetgen.consts import MERGE_NONE
from jetgen.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir, make_image_subdirs
)
from jetgen.utils.parsers import add_standard_eval_parsers

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions'
    )
    add_standard_eval_parsers(parser)

    parser.add_argument(
        '--domain',
        default = None,
        dest    = 'domain',
        help    = 'source domain  0 -- noise, 1 -- real data',
        type    = int
    )

    parser.add_argument(
        '--subsample',
        choices = [ 'linear-ddpm', 'linear-ddim', ],
        default = None,
        dest    = 'subsample',
        help    = 'use subsampling',
        type    = str,
    )

    parser.add_argument(
        '--sn',
        dest    = 'n_subsample',
        help    = 'number of subsampling steps',
        type    = int,
    )

    parser.add_argument(
        '-s', '--seed',
        default = 0,
        dest    = 'seed',
        help    = 'dp seed',
        type    = int,
    )

    parser.add_argument(
        '--dp',
        choices = [ 'ddpm', 'ddim', ],
        default = None,
        dest    = 'dp',
        help    = 'diffusion process to use',
        type    = str,
    )

    return parser.parse_args()

def save_np_array(arr, savedir, name, index):
    root = os.path.join(savedir, name)
    os.makedirs(root, exist_ok = True)

    path = os.path.join(root, f'sample_{index}.npz')
    np.savez_compressed(path, np.squeeze(arr))

def save_data(model, savedir, sample_counter):
    for (name, torch_image) in model.get_images().items():
        if torch_image is None:
            continue

        for index in range(torch_image.shape[0]):
            sample_index = sample_counter[name]
            image        = tensor_to_image(torch_image[index])

            save_np_array(image, savedir, name, sample_index)
            sample_counter[name] += 1

def evaluate_dp_single_domain(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter,
    diff_kwargs
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)

    for batch in tqdm.tqdm(data_it, desc = f'Dumping {domain}', total = steps):
        model.set_input(batch, domain = domain)
        model.diffuse(**diff_kwargs)

        save_data(model, savedir, sample_counter)

def evaluate_dp(
    model, domain_data_list, n_eval, batch_size, savedir, diff_kwargs
):
    # pylint: disable=too-many-arguments
    make_image_subdirs(model, savedir)
    sample_counter = collections.defaultdict(int)

    for domain, data_it in domain_data_list:
        evaluate_dp_single_domain(
            model, data_it, domain, n_eval, batch_size, savedir,
            sample_counter, diff_kwargs
        )

def main():
    cmdargs = parse_cmdargs()

    diff_kwargs = { 'dp' : cmdargs.dp, }
    if cmdargs.subsample is not None:
        diff_kwargs['subsample'] = {
            'name' : cmdargs.subsample,
            's'    : cmdargs.n_subsample,
        }

    args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type = MERGE_NONE, domain = cmdargs.domain
    )

    if cmdargs.domain is not None:
        domain_data_list = [ (cmdargs.domain, data_list), ]
    else:
        domain_data_list = list(enumerate(data_list))

    if cmdargs.seed is not None:
        model.reseed(cmdargs.seed)

    subdir = f'diffusion_dp({cmdargs.dp})-s({cmdargs.seed})'

    if cmdargs.subsample is not None:
        subdir += f'-sub({cmdargs.subsample}:{cmdargs.n_subsample})'

    savedir = get_eval_savedir(
        evaldir, subdir, cmdargs.model_state, cmdargs.split
    )

    evaluate_dp(
        model, domain_data_list, cmdargs.n_eval, args.batch_size, savedir,
        diff_kwargs
    )

if __name__ == '__main__':
    main()

