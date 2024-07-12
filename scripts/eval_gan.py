#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np

from jetgen.consts import MERGE_NONE
from jetgen.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir
)
from jetgen.utils.parsers import add_standard_eval_parsers

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Evaluate GAN')
    add_standard_eval_parsers(parser)

    parser.add_argument(
        '--domain',
        default = None,
        dest    = 'domain',
        help    = 'source domain  0 -- noise, 1 -- real data',
        type    = int
    )

    return parser.parse_args()

def save_images(model, savedir, sample_counter):
    for (name, torch_image) in model.get_images().items():
        if torch_image is None:
            continue

        if torch_image.ndim < 4:
            continue

        root = os.path.join(savedir, name)
        os.makedirs(root, exist_ok = True)

        for index in range(torch_image.shape[0]):
            sample_index = sample_counter[name]

            image = tensor_to_image(torch_image[index])
            path  = os.path.join(root, f'sample_{sample_index}.npz')

            sample_counter[name] += 1
            np.savez_compressed(path, np.squeeze(image))

def evaluate_gan_single_domain(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
    desc = f'Evaluating GAN domain {domain}'

    for batch in tqdm.tqdm(data_it, desc = desc, total = steps):
        model.set_input(batch, domain = domain)
        model.forward_nograd()

        save_images(model, savedir, sample_counter)

def evaluate_gan(model, domain_data_list, n_eval, batch_size, savedir):
    sample_counter = collections.defaultdict(int)

    for domain, data_it in domain_data_list:
        evaluate_gan_single_domain(
            model, data_it, domain, n_eval, batch_size, savedir, sample_counter
        )

def main():
    cmdargs = parse_cmdargs()

    args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type = MERGE_NONE, domain = cmdargs.domain
    )

    if cmdargs.domain is not None:
        domain_data_list = [ (cmdargs.domain, data_list), ]
    else:
        domain_data_list = list(enumerate(data_list))

    savedir = get_eval_savedir(
        evaldir, 'gan_eval', cmdargs.model_state, cmdargs.split
    )

    evaluate_gan(
        model, domain_data_list, cmdargs.n_eval, args.batch_size, savedir
    )

if __name__ == '__main__':
    main()

