#!/bin/bash
#SBATCH --job-name=NLP_train_text_generator
#SBATCH --partition=any
#SBATCH --nodes=1              # number of nodesGB
#SBATCH --mem=180GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=2    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/smuthi2s/perl5/NLP/Image_Storyteller/word_embeddings/logs/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/smuthi2s/perl5/NLP/Image_Storyteller/word_embeddings/logs/job_tf.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/tf-2

# locate to your root directory
cd /home/smuthi2s/perl5/NLP/Image_Storyteller/word_embeddings

# run the script
python train.py

