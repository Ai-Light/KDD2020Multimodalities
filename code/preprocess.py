import pandas as pd 
import numpy as np
import json
import base64
import swifter
from tqdm import tqdm
import csv
import pickle
from sklearn.externals import joblib
import gc
from time import sleep


TRAIN_PATH = '../data/train.tsv'
VAL_PATH = '../data/valid.tsv'
VAL_ANS_PATH = '../data/valid_answer.json'
SAMPLE_PATH = '../data/train.sample.tsv'
LABEL_PATH = '../data/multimodal_labels.txt'
TEST_PATH = '../data/testA.tsv'

def get_label(path):
    with open(path) as f:
        lines = f.readlines()
        label2id = {l.split('\n')[0].split('\t')[1]:int(l.split('\n')[0].split('\t')[0]) for l in lines[1:]}
        id2label = {int(l.split('\n')[0].split('\t')[0]):l.split('\n')[0].split('\t')[1] for l in lines[1:]}
    return label2id, id2label

label2id, id2label = get_label(LABEL_PATH)

print(id2label, label2id)

def convertBoxes(num_boxes, boxes):
    return np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
def convertFeature(num_boxes, features,):
    return np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
def convertLabel(num_boxes, label):
    return np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
def convertLabelWord(num_boxes, label):
    temp = np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
    return '###'.join([id2label[t] for t in temp])
def convertPos(num_boxes, boxes, H, W):
    pos_list = []
    for i in range(num_boxes):
        temp = boxes[i]
        pos_list.append([temp[0]/W, 
                         temp[2]/W, 
                         temp[1]/H, 
                         temp[3]/H, 
                         ((temp[2] - temp[0]) * (temp[3] - temp[1]))/ (W*H),])
    return pos_list

# 读10000条训练数据    
train = pd.read_csv(TRAIN_PATH,sep='\t', chunksize=10000, nrows = 10000, quoting=csv.QUOTE_NONE)
LEN = 0
product_set = set()
num_boxes_list = []
image_h_list = []
image_w_list = []
words_len_list = []
words_list = []
label_list = []
label_words_list = []
boxes_list = []
boxes_feature_list = []
pos_list = []

i = 0
for t in tqdm(train):
    print("starting")
    gc.collect()
    sleep(1)
    LEN += len(t)
    temp = list(t['query'])
    words_len_list.extend([len(q.split()) for q in temp])
    words_list.extend(temp)
    t['labels_convert_words'] = t.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    temp = list(t['labels_convert_words'])
    label_words_list.extend(temp)
    t['boxes_convert'] = t.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    temp = list(t['boxes_convert'])
    boxes_list.extend(temp)
    t['feature_convert'] = t.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    temp = list(t['feature_convert'])
    boxes_feature_list.extend(temp)
    t['pos'] = t.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
    temp = list(t['pos'])
    pos_list.extend(temp)
    del temp
    gc.collect()
    sleep(60)
    i += 1
    
print(LEN, len(product_set))

data = pd.DataFrame({
                     'words':words_list,
                     'label_words':label_words_list,
                     'features':boxes_feature_list,
                     'pos':pos_list,
                    })
print(data.head(10))
with open('../data/temp_data.pkl', 'wb') as outp:
    joblib.dump(data, outp)
print("temp data finish")

val = pd.read_csv(VAL_PATH,sep='\t')
val['boxes_convert'] = val.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
val['feature_convert'] = val.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
val['labels_convert'] = val.swifter.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
val['label_words'] = val.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
val['pos'] = val.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
del val['boxes'], val['features'], val['class_labels']    
with open('../data/val_data.pkl', 'wb') as outp:
    pickle.dump(val, outp)             
print("val data finish")

test = pd.read_csv(TEST_PATH,sep='\t')
test['boxes_convert'] = test.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
test['feature_convert'] = test.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
test['labels_convert'] = test.swifter.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
test['label_words'] = test.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
test['pos'] = test.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
del test['boxes'], test['features'], test['class_labels']
with open('../data/test_data.pkl', 'wb') as outp:
    pickle.dump(test, outp)
print("test data finish")
