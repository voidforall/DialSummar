from reader.multiwoz import MultiwozReader
from model.mtModel import SPNet
import json

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.common import Params
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.attention import DotProductAttention
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

import torch
import torch.optim as optim
from utils.predictor import MultiwozPredictor

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
model = SPNet(vocab, source_embedder, encoder1, encoder2, attention)

print("By default, we apply our pretrained weights to the model.")
with open("./models_saved/model_weights.th", 'rb') as f:
    model.load_state_dict(torch.load(f))

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
    
predictor = MultiwozPredictor(model, dataset_reader=reader)

print("In the inference, the model predicts the summary of the 1000 test cases.")
print("The result will be saved in ./predictions.txt.")

test_path = "./data/test_delex.json"
test_json = json.load(open(test_path))
test_cases = [v for v in test_json.values()]

predictor = MultiwozPredictor(model, dataset_reader=reader)
prediction_list = []
for i in range(1000):
    test_case = test_cases[i]
    prediction = " ".join(predictor.predict_json(test_case)["predicted_tokens"])
    prediction_list.append(prediction)
    
pred_file = open("./predictions.txt", "w")
for i in range(1000):
    pred_file.write(" ")
    pred = prediction_list[i] + " \n"
    pred_file.write(pred)

