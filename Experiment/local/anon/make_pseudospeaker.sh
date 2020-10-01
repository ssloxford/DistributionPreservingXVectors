#!/bin/bash

# Modified Work Copyright (C) 2020 <Henry Turner, Giulio Lovisotto, Ivan Martinovic>

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

. path.sh
. cmd.sh

rand_level="spk"
cross_gender="false"

rand_seed=2020
combine_genders="false"
pca_size=20
gmm_size=3
threshold=1.01

stage=0

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: "
  echo "  $0 [options] <src-data-dir> <pool-data-dir> <xvector-out-dir> <plda-dir>"
  echo "Options"
  echo "   --rand-level=spk     # [utt, spk] Level of randomness while computing the pseudo-xvectors"
  echo "   --rand-seed=<int>     #  Random seed while computing the pseudo-xvectors"
  echo "   --cross-gender=true     # [true, false] Whether to select same or
                                                   other gender while computing the pseudo-xvectors"
  echo "   --combine-genders=false # Separate GMMs for both genders or not"
  echo "   --pca-size=<int or float>  #  int for number of PCA components, percentage for proportion of variance to be captured"
  echo "   --gmm-size=<int>     # Number of components per mixture model"
  echo "   --threshold=<float>     # Similarity threshold. New xvecs must have cosine similarity below this value with the original xvector for this speaker"
  exit 1;
fi

src_data=$1
pool_data=$2
xvec_out_dir=$3
plda_dir=$4
src_dataname=$(basename $src_data)
pool_dataname=$(basename $pool_data)
src_xvec_dir=${xvec_out_dir}/xvectors_${src_dataname}
pool_xvec_dir=${xvec_out_dir}/xvectors_${pool_dataname}
pseudo_xvecs_dir=${src_xvec_dir}/pseudo_xvecs

mkdir -p ${pseudo_xvecs_dir}

src_spk2gender=${src_data}/spk2gender
pool_spk2gender=${pool_data}/spk2gender



if [ $stage -le 0 ]; then
  # Create a GMM for each gender, and then sample randomly from GMM to provide new plausible xvectors
  echo "Generating Pseudo Xvectors"
  python local/anon/gen_pseudo_xvecs.py ${src_data} ${pool_data} \
	  ${xvec_out_dir} ${pseudo_xvecs_dir} ${src_xvec_dir} ${rand_level} \
    ${cross_gender} ${rand_seed} --pca_size ${pca_size} --gmm_size ${gmm_size} \
     --threshold ${threshold} --combine_genders ${combine_genders} --pickle_file data/models.pickle || exit 1;
fi

