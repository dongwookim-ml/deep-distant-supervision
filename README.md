# deep-distant-supervision
Distant supervision algorithms with neural network.

# Model

## Training
To train the deep-distant supervision model, try

`python main.py --is_train=True --num_hidden=256 --data_set=nyt --num_poch=3 --save_gap=1000 --print_gap=100`

Examples of some configurable model parameters are:
- `word_attn`: boolean, whether to use word-level attention layer
- `sent_attn`: boolean, whether to use sentence-level attention layer
- `bidirectional`: boolean, whether to use bidirectional rnn 
- `data_set`: path to dataset folder. the folder should contain `train_x.npy` and `train_y.npy` for training
- `save_gap`: the number of batch steps used to save partially trained models.
- `print_gap`: print status of training for every `print_gap`

To see the all possible configurations, try

`python main.py --help`

### Data format
The model requires input file, predefined set of relations, and precomputed word-embedding vectors.
Take a look at the following files for the required input format:
- `data/nyt/train.txt`: each line consists of (entity1_id, entity2_id, entity1_surface_form, entity2_surface_form, relation, sentence)
- `data/nyt/relation2id`: list of relations with their ids
- `data/word2vec.txt`: each line consists of (word_token embedding_vector)

## Testing
To test the trained model, try

`python main.py --is_train=False --data_set=nyt`

# Requirements
- Tensorflow 1.2
- numpy, tqdm, scikit-learn

# References
- https://github.com/thunlp/TensorFlow-NRE
