# Track 1: Counter Argument Generation

## 0. Set Up

### 0.1 Dataset

The dataset is a txt format file, and each line includes the topic discussed, the original argument and the counter argument.

Please refer to the [official website](http://www.fudan-disc.com/sharedtask/AIDebater23/index.html) for competition registration and dataset downloading. All data files should be put in `data/`.

### 0.2 Requirements

- tqdm
- numpy
- pytorch
- rouge
- time
- transformers

### 1. Training

We provide a baseline model fine-tuned on `gpt2`. You can train the baseline model by running

```{bash}
python train.py
```

The model will only be fine-tuned for one epoch. And then it will be saved in `model/`.

### 2. Evaluation

To evaluate the trained model, you can simply run

```{bash}
python eval.py
```

The program will report performance metrics like ROUGE-L.
