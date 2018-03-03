# Generative-Discriminative-Persona-Classification-Model

## Quick Start

For **Persona Classification** task, we design **Traditional Machine Learning Model** as  baseline. Three main models have been implemented in our project based on [IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq), which are:

- **Discriminative Model**
- **Generative Model**
- **Hierarchical Model**

You can enter one of these folders, and type `python main.py` in command line, default parameters are set carefully by us.

Also, you can use ```python main.py -h```  to check out more details:

```shel
  -h, --help                                show this help message and exit
  --path PATH                               Input data path.
  --num_sentence NUM_SENTENCE               Number of sentences in every dialog
  --max_len MAX_LEN                         Choose max_len of Encoder
  --hidden_size HIDDEN_SIZE                 Choose the hidden size
  --num_spk NUM_SPK                         Choose the speaker size
  --spk_embed_size SPK_EMBED_SIZE           Define the speaker embedding size
  --cuda CUDA                               Use cuda or not
  --lr LR                                   Choose the learning rate
  --batch_size BATCH_SIZE                   Choose the batch_size for train, dev and test
```

## Models

### Discriminative/Generative/Hierarchical Model

- **main.py:**  to parse arguments and launch experiments
- **load_model.py:** resume checkpoints for continuing training or parameters tuning
- **seq2mlp & seq2seq:**  folders for model implementation
  - **dataset/:**  scripts for reading and parsing the corpus
  - **models/:** networks via pytorch, such as EncoderRNN, DecoderRNN and Seq2Seq 
  - **loss/:** overwrite nn.Loss, nn.NLLLoss in pytorch, for bidirectional loss computing
  - **optim/:** optimaters for networks
  - **evaluator/:** evaluator and predictor for networks
  - **util/:** checkpoint saver

### Traditional Machine Learning Model

- **tfidf.py:** traditional models for tfidf + LogisticRegression / RandomForest / Naive Bayes
- **metric.py:** confusion metrix generation for traditional models

## Corpus

We use the text in **Big Theory** Season 1 - 9 remaining 6 main characters and regarding other characters as 'Others' category. We bring two data formats:

- **.tsv:** {context} \t {speaker} \t {response} 
- **.csv:** {speaker}, {context}, {speaker}, {context}, …, {speaker}, {response} 

They are in **data/** folder, and we also provide data pre-processor to convert your own raw data into these two formats, which are in **data/utils/** folder:

- **data/utils/**
  - **split.py:** split the raw text as  8:1:1 for every scene
  - **parse_raw.py:** convert raw text to csv format data with tokenization 
  - **csv2tsv.py:** convert csv format data into tsv format
  - **tsv2csv.py:** convert tsv format data into csv format
  - **augmentation.py:** use slide window to generate dialogs like 1234, 2345, 3456 instead of  1234, 5678
  - **clip.ipynb:** clip the too short and too long dialogs

## Visualization

- **visual_speaker.py:**  speaker embedding visualization
- **plot_confusion_matrix.py** 

## Environment

- Python 2.7
- Pytorch 0.3.0+ with CUDA 8.0 / CPU
- Pytorchtext 0.2.1

Or create a virtual env and use  `pip install -r requirements.txt`  to install them.

## Partners

- [AuCson](https://github.com/AuCson)
- [RshCaroline](https://github.com/Rshcaroline)
