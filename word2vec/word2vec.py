import numpy as np
import random
import os
import urllib
import urllib.request
import zipfile
import collections
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim


DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
DATA_FOLDER = "data"
FILE_NAME = "text8.zip"
EXPECTED_BYTES = 31344016

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
    
def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    local_file_path = os.path.join(DATA_FOLDER, file_name)
    if os.path.exists(local_file_path):
        print("Dataset ready")
        return local_file_path
    file_name, _ = urllib.request.urlretrieve(os.path.join(DOWNLOAD_URL, file_name), local_file_path)
    file_stat = os.stat(local_file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception(
              'File ' + file_name +
              ' might be corrupted. You should try downloading it with a browser.')
    return local_file_path

# Read the data into a list of strings.
def read_data(file_path):
    """ Read data into a list of tokens """
    with zipfile.ZipFile(file_path) as f:
        data = f.read(f.namelist()[0]).split()
    return data

# Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
    """ Create two dictionaries and count of occuring words
        - word_to_id: map of words to their codes
        - id_to_word: maps codes to words (inverse word_to_id)
        - count: map of words to count of occurrences
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    word_to_id = dict() # (word, id)
    # record word id
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys())) # (id, word)
    return word_to_id, id_to_word, count

def convert_words_to_id(words, dictionary, count):
    """ Replace each word in the dataset with its index in the dictionary """
    data_w2id = []
    unk_count = 0
    for word in words:
        # return 0 if word is not in dictionary
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data_w2id.append(index)
    count[0][1] = unk_count
    return data_w2id, count

# utility function
def generate_sample(center_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for idx, center in enumerate(center_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in center_words[max(0, idx - context) : idx]:
            x1.append(center)
            y1.append(target)
        # get a random target after the center word
        for target in center_words[idx + 1 : idx + context + 1]:
            x2.append(center)
            y2.append(target)
    x1.extend(x2)
    y1.extend(y2)
    return x1,y1

def batch_generator(data, skip_window, batch_size):
    """ Group a numeric stream into batches and yield them as Numpy arrays. """
    single_gen = generate_sample(data, skip_window)
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1], dtype=np.int32)
        for idx in range(batch_size):
            center_batch[idx], target_batch[idx] = next(single_gen)
        yield center_batch, target_batch

def neg_sample(num_samples, positives=[]):
    freqs_pow = torch.Tensor([freqs[id_to_word[i]] for i in range(vocabulary_size)]).pow(0.75)
    dist = freqs_pow / freqs_pow.sum()
    w = np.random.choice(len(dist), (len(positives), num_samples), p=dist.numpy())
    return torch.cuda.LongTensor(w)

class skip_gram(nn.Module):
    def __init__(self):
        super(skip_gram,self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size,embed_size)
        self.embedding_layer.weight.data.uniform_(-1,1)
    def forward(self,x,label):
        negs = neg_sample(5, label)
        u_embeds = self.embedding_layer(label).view(len(label), -1)
        v_embeds_pos = self.embedding_layer(x).mean(dim=1)
        v_embeds_neg = self.embedding_layer(negs).mean(dim=1)
        #print(u_embeds.size())
        #print(v_embeds_pos.size())
        #print(v_embeds_neg.size())

        loss1 = torch.diag(torch.matmul(u_embeds, v_embeds_pos))
        loss2 = torch.diag(torch.matmul(u_embeds, v_embeds_neg.transpose(0, 1)))
        loss1 = -torch.log(1 / (1 + torch.exp(-loss1)))
        loss2 = -torch.log(1 / (1 + torch.exp(loss2)))
        loss = (loss1.mean() + loss2.mean())
        return(loss) 
    
make_dir(DATA_FOLDER)
file_path = download(FILE_NAME, EXPECTED_BYTES)
vocabulary = read_data(file_path)

vocabulary_size = 50000
word_to_id, id_to_word, count = build_dataset(vocabulary, vocabulary_size)
data_w2id, count = convert_words_to_id(vocabulary, word_to_id, count)
freqs = Counter(vocabulary)

del vocabulary  # reduce memory.
#print('Most common words (+UNK)', count[:5])
#print('Sample data: {}'.format(data_w2id[:10]))
#print([id_to_word[i] for i in data_w2id[:10]])

## some training settings
training_steps = 100000
skip_step = 2000
ckpt_dir = "checkpoints/word2vec_simple"

## some hyperparameters
batch_size = 128
embed_size = 128
num_sampled = 64
learning_rate = 1.0

X,Y = generate_sample(data_w2id,2)
print("Data_container ready")

tensor_X = torch.cuda.LongTensor(X)
tensor_Y = torch.cuda.LongTensor(Y)
torch_dataset = torch.utils.data.TensorDataset(tensor_X,tensor_Y)
print("Torch dataset ready")

data_iter = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                        shuffle=False)
model = skip_gram()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):
    total_loss = torch.Tensor([0])
    for input_data, target in data_iter:
        model.zero_grad()
        target = target.cuda()
        loss = model(input_data, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print('epoch %d loss %.4f' %(epoch, total_loss))
print(losses)





