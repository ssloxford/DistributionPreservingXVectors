# Distribution Preserving X-Vectors for Speaker Anonymization
### An entry in the Voice Privacy Challenge 2020
This repository contains the code for the paper [Speaker Anonymization with Distribution-Preserving X-Vector Generation for the Voice Privacy Challenge 2020](www.fixlink.com) presented at the [Voice Privacy Challenge Special Session at Interspeech 2020](https://www.voiceprivacychallenge.org/).
This work is a collaboration between [Henry Turner](https://www.cs.ox.ac.uk/people/henry.turner/) and [Giulio Lovisotto](https://github.com/giuliolovisotto/) from the [System Security Lab](https://seclab.cs.ox.ac.uk/) at University of Oxford.


## Idea
The idea for our challenge entry began by examining the similarity distributions of x-vectors from anonymized voices generated by the baseline system, with those from organic voices, which revealed that the anonymized voices had significantly higher inter voice similarity, as can be seen in the following density plots of the similarity scores:
<p align="center"><img src="/github_images/baseline_comparison.png" width="50%"></p>

<img src="/github_images/TSNEMales.png" width="30%" align="left"> <br><br><br> We examined this further with a t-SNE analysis, conducted on a set comprised of half anonymized x-vectors and half organic x-vectors. This t-SNE analysis, shown below, clearly highlights the differences that exist between the two types of x-vectors, with the fake ones being more similar to one another and less diverse than organic x-vectors.


We set out to remedy this, by developing a technique to generate psuedo-xvectors that better utilize the x-vector hyperspace, thus creating more diverse x-vectors and as a result voices anonymized voices that are more distinct from another.

We do this by training a PCA decomposition model and a Gaussian Mixture Model (GMM) fitted on a sample of organic x-vectors that have been transformed into the PCA space.
We can then generate samples from this GMM, which we apply an inverse PCA to, giving us an x-vector for that speaker.
We then use the same voice processing pipeline as the baseline solution to generate an anonymized voices, which replaces the speakers x-vector with the newly created anonymized x-vector, as shown in the diagram below.
<p align="center"><img src="/github_images/sys_diag.png" width="70%"></p>

Please see our paper [here for further details](www.arxivlinkhere)




<!-- <p align="center"><img src="/github_images/transferability.png" width="80%"></p> -->

<!-- ### Resources
Add paper, videos when available
) -->

<!-- ## Citation
If you use this repository please cite the paper as follows:
```
TODO citation here
``` -->
## Contributors
 * [Giulio Lovisotto](https://github.com/giuliolovisotto/)
 * [Henry Turner](https://www.cs.ox.ac.uk/people/henry.turner/)

## Acknowledgements

This work was generously supported by a grant from Master-card  and  by  the  Engineering  and  Physical  Sciences  ResearchCouncil \[grant numbers EP/N509711/1, EP/P00881X/1\]
 

## About the code
The system is based on the Voice-Privacy-Challenge 2020 baseline system, which can be found on Github [here](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020).


A Dockerfile is provided, which will clone the above repository and install Kaldi.

As part of running the recipe the models used for evaluation must be downloaded from the Voice Privacy Challenge organisers (stage 1). This currently requires a password from them, and the process for getting this is described on the Github linked above. After the challenge has fully completed I will ask about including a download link separately in this code base.

## Running the code

### Requirements
- Docker installation, with nvidia gpu support
- Docker compose installed

### Run Voice Privacy Challenge Experiments
- run `docker-compose up` in the Experiment folder N.B this can take a while, as it builds Kaldi and runs some installs which may take a while.
- attach to the container once it is built: `docker attach DistPrevXvecs`
- run the code with `./run.sh`


### Without running all Experiments
If you wish to only create the xvector generator run `train_models.sh` and then see `local/anon/gen_pseudo_xvecs.py` for its usage to create fake x-vectors.

To actually turn the fake x-vectors into audio see `local/anon/anonymize_data_dir.sh`, which will run the anonymization on a directory in Kaldi format.

