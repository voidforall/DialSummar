from reader.multiwoz import MultiwozReader
from model.mtModel import SPNet

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.common import Params
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.attention import DotProductAttention
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

import torch
import torch.optim as optim

reader = MultiwozReader()
print("Reading the delexiclaized data from training set and validation set...")
train_dataset = reader.read("./data/train_delex.json")
valid_dataset = reader.read("./data/valid_delex.json")

print("Building vocabulary from training set and validation set...")
vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
print("Temporary vocabulary has been built.")

params = Params({"token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 128
        }}})
EMBEDDING_DIM = 128
source_embedder = BasicTextFieldEmbedder.from_params(vocab, params=params)

HIDDEN_DIM = 256
encoder1 = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, bidirectional=True, batch_first=True))
encoder2 = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, bidirectional=True, batch_first=True))
attention = DotProductAttention()

print("Use SPNet with default setting...")
model = SPNet(vocab, source_embedder, encoder1, encoder2, attention)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
optimizer = optim.Adam(model.parameters(), lr=0.001)

iterator = BucketIterator(batch_size=8, sorting_keys=[("target_tokens", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=valid_dataset,
                  shuffle=True,
                  patience=5,
                  num_epochs=50,
                  summary_interval=100, # to tensorboard
                  serialization_dir = "./models_saved/SPNet/",
                  num_serialized_models_to_keep = 5,
                  grad_norm=2.0,
                  cuda_device=cuda_device)

print("The training starts, results will be serialized to dir", serialization_dir)
trainer.train()




