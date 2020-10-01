# Distribution Preserving X-Vectors for Speaker Anonymization




## About the code
The system is based on the Voice-Privacy-Challenge 2020 baseline system, which can be found on Github [here](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020).



A Dockerfile is provided, which will clone the above repository and install Kaldi.

As part of running the recipe the models used for evaluation must be downloaded from the Voice Privacy Challenge organisers (stage 1). This currently requires a password from them, and the process for getting this is described on the Github linked above. After the challenge has fully completed I will ask about including a download link separately in this code base.

## Running the code

### Requirements
- Docker installation, with nvidia gpu support
- Docker compose installed

### Run
- run `docker-compose up` in the Experiment folder
- attach to the container once it is built: `docker attach DistPrevXvecs`
- run the code with `./run.sh`
