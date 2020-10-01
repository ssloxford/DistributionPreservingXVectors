
# Significantly changed files
The following list of files explain where changes have occured from the Voice Privacy Challenge Baseline (xvector) system, and roughly what these changes are.

- `run.sh` ammended to contain all the parameters for calling our anonymization routine and passing the correct parameters through
- `train_models.sh` script just to create the xvector generator. Saves this with specified file name, can then be applied to 
- `local/anon/anonymize_data_dir.sh` mostly parameter passing changes
- `local/anon/make_pseudospeaker.sh` changes to call our pseudo speaker generation function
- `local/anon/gen_pseudo_xvecs.py` calls the code to train pca and gmm on the pool if they do not exist. Uses xvector generator to generate pseudo xvectors for each speaker passed in
- `local/make_voxceleb_spk2gender.py` created to make the spk2gender files for voxceleb correctly and resolve some issues that existed in these

