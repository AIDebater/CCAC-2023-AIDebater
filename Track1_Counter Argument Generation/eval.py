import torch
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from model import cag_gpt_model


model = cag_gpt_model(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
rouge = Rouge()
model.from_pretrained('./model')

with open('./data/test.txt', 'r', encoding = 'UTF-8') as f:
    lines = f.readlines()

scores = []
for line in tqdm(lines):
    topic, source, reference = line.strip('\n').split('\t')
    hypothesis = model.generator(topic, source, num_beams = 3, max_length = 128)
    scores.append(rouge.get_scores(reference, hypothesis, avg=True)["rouge-l"]['r'])
print('ROUGE-L score:{:.2f}'.format(np.mean(scores)))


