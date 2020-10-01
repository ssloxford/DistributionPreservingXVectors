
# Copyright (C) 2020 <Henry Turner, Giulio Lovisotto, Ivan Martinovic>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


import sys
from os.path import basename, join
import os
import operator
import argparse
import numpy as np
import random
from kaldiio import WriteHelper, ReadHelper
from sklearn import decomposition
from sklearn import metrics
from sklearn import mixture
from sklearn.metrics.pairwise import cosine_similarity
import pickle

"""
Custom version of gen_pseudo_xvecs which adds the ability to load a gmm and pca
"""

def generate_xvector(transforms, original_xvector=None, distance_threshold=1.01, iter_limit=1000):
    gmm = transforms['gmm']
    pca = transforms['pca']
    
    cosine_distance = np.inf
    min_cos_dist = np.inf
    smallest_xvec = None
    iter_count = 0
    # voices are too similar if the cos similarity above threshold, so we recurse to try again.
    while cosine_distance > distance_threshold and iter_count < iter_limit:
        sample, _ = gmm.sample()
        xvec = pca.inverse_transform(sample)
        cosine_distance = cosine_similarity(xvec, original_xvector.reshape(1, -1)) if original_xvector is not None else 0
        if cosine_distance < min_cos_dist:
            min_cos_dist = cosine_distance
            smallest_xvec = xvec
        iter_count += 1

    if iter_count == iter_limit:
        return smallest_xvec
    return xvec


def generate_pca_and_gmm(xvectors, pca_parameter, random_state=None):
    xvectors = np.array(xvectors)
    pca = decomposition.PCA(pca_parameter, whiten=False,
                            random_state=random_state)
    pca.fit(xvectors)
    pca_xvecs = pca.transform(xvectors)
    max_iter = 1000
    tol = 1e-15
    gmm = mixture.GaussianMixture(args.gmm_size, covariance_type='diag', tol=tol, init_params="kmeans", n_init=1, max_iter=max_iter, random_state=random_state)
    gmm.fit(pca_xvecs)
    fail_counter = 0
    while not gmm.converged_:
        fail_counter+=1
        tol=tol*2
        max_iter = int(max_iter*1.1)
        gmm = mixture.GaussianMixture(args.gmm_size, covariance_type='diag', tol=tol, init_params="random", n_init=2, max_iter=max_iter)
        gmm.fit(pca_xvecs)

    return {'pca': pca, 'gmm': gmm}


def load_pca_and_gmm(pickle_file):
    print("Loading existing models:{}".format(pickle_file))
    return pickle.load(open(pickle_file, 'rb'))

def save_pca_and_gmm(transforms, pickle_file):
    pickle.dump(transforms, open(pickle_file, 'wb'))


def train_models(pool_data, xvec_out_dir, combine_genders=False):
     # Load and assemble all of the xvectors from the pool sources
    pool_data_sources = os.listdir(pool_data)
    pool_data_sources = [x for x in pool_data_sources if os.path.isdir(
        join(pool_data, x)) and os.path.isfile(os.path.join(pool_data, x, 'wav.scp'))]

    gender_pools = {'m': [], 'f': []}
    xvector_pool = []

    for pool_source in pool_data_sources:
        print('Adding {} to the pool'.format(join(pool_data, pool_source)))
        pool_spk2gender_file = join(pool_data, pool_source, 'spk2gender')

        # Read pool spk2gender
        pool_spk2gender = {}
        with open(pool_spk2gender_file) as f:
            for line in f.read().splitlines():
                sp = line.split()
                pool_spk2gender[sp[0]] = sp[1]

        # Read pool xvectors
        pool_xvec_file = join(xvec_out_dir, 'xvectors_'+pool_source,
                              'spk_xvector.scp')
        if not os.path.exists(pool_xvec_file):
            raise ValueError(
                'Xvector file: {} does not exist'.format(pool_xvec_file))

        with ReadHelper('scp:'+pool_xvec_file) as reader:
            for key, xvec in reader:
                # print key, mat.shape
                xvector_pool.append(xvec)
                gender = pool_spk2gender[key]
                gender_pools[gender].append(xvec)

    print("Read ", len(gender_pools['m']), " male pool xvectors")
    print("Read ", len(gender_pools['f']), " female pool xvectors")

    # Fit and train GMMS
    if combine_genders:
        transforms = generate_pca_and_gmm(
            xvector_pool, pca_parameter, random_state=random_seed)
    else:
        transforms = {'m': {}, 'f': {}}
        for gender in ('m', 'f'):
            gender_xvecs = gender_pools[gender]
            transforms[gender] = generate_pca_and_gmm(
                gender_xvecs, pca_parameter, random_state=random_seed)
    
    return transforms

def load_src_spk_files(src_data):
    # assign new xvectors
    src_spk2gender_file = join(src_data, 'spk2gender')
    src_spk2utt_file = join(src_data, 'spk2utt')

    # Read source spk2gender and spk2utt
    src_spk2gender = {}
    src_spk2utt = {}
    print("Reading source spk2gender.")

    if not os.path.exists(src_spk2gender_file):
        raise ValueError("{} does not exist!".format(src_spk2gender_file))
    with open(src_spk2gender_file) as f:
        for line in f.read().splitlines():
            sp = line.split()
            src_spk2gender[sp[0]] = sp[1]
    print("Reading source spk2utt.")

    if not os.path.exists(src_spk2utt_file):
        raise ValueError("{} does not exist!".format(src_spk2utt_file))

    with open(src_spk2utt_file) as f:
        for line in f.read().splitlines():
            sp = line.split()
            src_spk2utt[sp[0]] = sp[1:]

    return src_spk2gender, src_spk2utt

def write_new_xvectors(pseudo_xvecs_dir, pseudo_xvec_map):
    # Write features as ark,scp
    print("Writing pseud-speaker xvectors to: "+pseudo_xvecs_dir)
    ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
        pseudo_xvecs_dir, 'pseudo_xvector',
        pseudo_xvecs_dir, 'pseudo_xvector')
    with WriteHelper(ark_scp_output) as writer:
        for uttid, xvec in pseudo_xvec_map.items():
            writer(uttid, xvec)

def load_xvecs(xvec_file):
    original_xvecs = {}
    # Read source original xvectors.
    with ReadHelper('scp:' + xvec_file) as reader:
        for key, xvec in reader:
            # print key, mat.shape
            original_xvecs[key] = xvec
    return original_xvecs

def write_new_spk2gender(pseudo_xvecs_dir, pseudo_gender_map):

    print("Writing pseudo-speaker spk2gender.")
    with open(join(pseudo_xvecs_dir, 'spk2gender'), 'w') as f:
        spk2gen_arr = [spk+' '+gender for spk,
                    gender in pseudo_gender_map.items()]
        sorted_spk2gen = sorted(spk2gen_arr)
        f.write('\n'.join(sorted_spk2gen) + '\n')

def generate_new_xvectors(transforms, original_xvecs, src_spk2gender, cross_gender, threshold, random_level='spk'):
    # store the new xvector for the speaker
    pseudo_xvec_map = {}
    # store the gender of the new speaker
    pseudo_gender_map = {}
    for spk, gender in src_spk2gender.items():

        original_xvec = np.array(original_xvecs[spk])

        # If we are doing cross-gender VC, reverse the gender else gender remains same
        if cross_gender:
            gender_rev = {'m': 'f', 'f': 'm'}
            gender = gender_rev[gender]

        # the new gender of the speaker
        pseudo_gender_map[spk] = gender
        selected_transform = transforms if combine_genders else transforms[gender]

        if rand_level == 'spk':
            # For rand_level = spk, one xvector is assigned to all the utterances
            # of a speaker
            pseudo_xvec = generate_xvector(
                transforms=selected_transform, original_xvector=original_xvec, distance_threshold=threshold)
            # Assign it to all utterances of the current speaker
            for uttid in src_spk2utt[spk]:
                pseudo_xvec_map[uttid] = pseudo_xvec
        elif rand_level == 'utt':
            # For rand_level = utt, random xvector is assigned to all the utterances
            # of a speaker
            for uttid in src_spk2utt[spk]:
                # Compute random vector for every utt
                pseudo_xvec = generate_xvector(
                    transforms=selected_transform, original_xvector=original_xvec, distance_threshold=threshold)
                # Assign it to all utterances of the current speaker
                pseudo_xvec_map[uttid] = pseudo_xvec
        else:
            print("rand_level not supported! Errors will happen!")
    return pseudo_xvec_map, pseudo_gender_map


if __name__ == "__main__":
    print(sys.argv)
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('src_data')
    parser.add_argument('pool_data')
    parser.add_argument('xvec_out_dir')
    parser.add_argument('pseudo_xvecs_dir')
    parser.add_argument('src_xvec_dir')
    parser.add_argument('rand_level', choices=['utt', 'spk'])
    parser.add_argument('cross_gender', choices=['true', 'false'])
    parser.add_argument('random_seed', default=2020, type=int)
    parser.add_argument('--pca_size', default=0.90, type=float,
                        help='PCA proportion to be captured, as a percentage (float)')
    parser.add_argument('--gmm_size', default=2, type=int,
                        help='GMM number of components')
    parser.add_argument('--threshold', default=1.01, type=float,
                        help='Threshold to repeat random x vector if voice too similar')  # default set to 1.01 so its not applied as cosine limited from -1 to 1
    parser.add_argument('--combine_genders', choices=['true', 'false'], default='false',
                        help='Flag to construct just one GMM for both genders')
    parser.add_argument('--pickle_file', help="Pickle file location to support reuse. Use 'None' to disable or leave blank")

    args = parser.parse_args()
    src_data = args.src_data
    pool_data = args.pool_data
    xvec_out_dir = args.xvec_out_dir
    pseudo_xvecs_dir = args.pseudo_xvecs_dir
    src_xvec_dir = args.src_xvec_dir
    rand_level = args.rand_level
    cross_gender = True if args.cross_gender == 'true' else False
    random_seed = args.random_seed
    pca_parameter = args.pca_size
    combine_genders = True if args.combine_genders == 'true' else False

    pickle_file = args.pickle_file
    if pickle_file == 'None':
        pickle_file = None

    # N.B 1  is the score for very similar voices, so this is a less than threshold.
    threshold = args.threshold

    if pca_parameter.is_integer():
        pca_parameter = int(pca_parameter)

    if cross_gender:
        print("**Opposite gender speakers will be selected.**")
    else:
        print("**Same gender speakers will be selected.**")


    # are we training a new gmm or do we have an existing?
    if pickle_file is not None and os.path.exists(pickle_file):
        transforms = load_pca_and_gmm(pickle_file)
        if (not combine_genders) and ('m' not in transforms):
            raise ValueError('Loaded GMM was not compatible with the combine genders parameter used:{combine_genders}')

    else:
        transforms = train_models(pool_data, xvec_out_dir, combine_genders=combine_genders)
        if pickle_file is not None:
            pickle.dump(transforms, open(pickle_file, 'wb'))

    src_spk2gender, src_spk2utt = load_src_spk_files(src_data)

    original_xvecs = load_xvecs(xvec_file=src_xvec_dir + '/spk_xvector.scp')

    new_xvecs, new_genders = generate_new_xvectors(transforms, original_xvecs, src_spk2gender, cross_gender, threshold, rand_level)
    
    write_new_xvectors(pseudo_xvecs_dir, new_xvecs)
    write_new_spk2gender(pseudo_xvecs_dir, new_genders)

