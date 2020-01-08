# README

The code is a full implementation of `SPNet` in my paper `Abstractive Dialog Summarization with Semantic Scaffolds`, which is my first trial in independent research. Although the result is not satisfying, I still learn a lot from the experience.


### Environment

Our model is built in `AllenNLP`, which is an open-source NLP research library built on `PyTorch`. You can install it following its [GitHub repo](https://github.com/allenai/allennlp).



### Data

We adapt [MultiWOZ-2.0](https://github.com/budzianowski/multiwoz) to our abstractive dialog summarization task. We release the delexicalized data with domain annotation in `./data` folder. And`./reqt_dict.json` is the requests in test set, used for evaluation.



### Train

We use default setting of `SPNet` in `./models/mtModel` and `./train.py`, run `python3 train.py` to start training. Note that you may need to specify the GPU in the code. Logs and models will be saved in `./models_saved/SPNet`. We also provide pretrained weights in `./models_saved/model_weights.th`. 



### Inference

Run `python3 inference.py` and the predictor will generate `./predictions.txt` with 1,000 lines of summaries. They are the prediction of 1,000 test samples and the reference summaries are saved in `./references.txt`. Note that it will use our pre-trained model as a default choice.



### Evaluate

We have two automatic evaluation metrics: ROUGE and CIC. To separate the inference and evaluation, we use file2rouge tools to measure the ROUGE scores. You can follow the instructions in [files2rouge GitHub page](https://github.com/pltrdy/files2rouge).

CIC evaluation method is shown in `./evaluate.py`.


