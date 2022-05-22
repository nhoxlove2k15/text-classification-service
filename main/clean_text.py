# -*- coding: utf-8 -*-
import pandas as pd
from fairseq import *
import fastBPE
import argparse
path_bert = '../resource/phobert/'

# parser = argparse.ArgumentParser()
# parser.add_argument('--bpe-codes', 
#     default=path_bert + "bpe.codes",
#     required=False,
#     type=str,
#     help='path to fastBPE BPE'
# )
# args, unknown = parser.parse_known_args()
# print("args: ",args)
# print("unknow: ", unknown)
# bpe = fastBPE(args)
# print("bpe: ",bpe.bpe)

# # Load the dictionary
# vocab = Dictionary()
# vocab.add_from_file(path_bert + "dict.txt")
# print("vocab: ", vocab.count)
# stopwords
stopwords = []
with open("./thesis/vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
    for line in f :
      stopwords.append(line.replace('\n',''))
# print(stopwords)
data = pd.read_csv('./thesis/new_tag.csv')
tag1 = data["tag1"].tolist()
tag2 = data["tag2"].tolist()
# print(len(tag2))
# ['phòng thủ, tấn công']
# print(tag2[0].split(','))
def to_replace(text):
  for i in range(0,len(tag2) - 1):
      # print(tag2[i])
      for word in tag2[i].split(','):
        #  print(word.strip())
         text = text.replace(word.strip(), tag1[i])
  return text

from pyvi import ViTokenizer, ViPosTagger
import string

punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~""" # except '-'
def my_remove_line_and_punctuation(text):
    text = text.replace("\n", " ")
    new_text = text.translate(str.maketrans('', '', punctuation))
    return new_text

def my_remove_stopword(text):
    for i in stopwords:
      word = ' ' + i + ' '
      if word in text:
        text = text.replace( word ,' ')
    return text
def to_lower_case(text):
    return text.lower()
def tokenizer(text):
    token = ViTokenizer.tokenize(text)
    return token
  
def clean_text(text):
    print("1")
    text = my_remove_line_and_punctuation(text)
    print("2")
    text = to_lower_case(text)
    print("3")
  
    text = to_replace(text)
    print("4")

    text = my_remove_stopword(text)
    print("5")

    text = tokenizer(text)
    print("6")




    return text
tag_dict = dict
text = """"
Xây dựng chương trình điều khiển xe tự hành ứng dụng Deep Learning Đề tài xây dựng chương trình điều khiển xe tự hành ứng dụng Deep Learning
được chúng tôi chia thành hai bài toán nhỏ gồm xác định góc lái xe và xác định tín
hiệu giao thông, các tín hiệu giao thông gồm 15 loại biển báo và ba loại tín hiệu đèn
giao thông, cụ thể gồm: 224, 122, 301e, 301d, 301a, 102, 302a, 302b, 202, 127, 134,
408, 123a, 123b, 124a, tín hiệu đèn xanh, đèn đỏ, đèn vàng. Đề tài được thực hiện
trên xe mô hình và đường mô hình. Đề tài sử dụng hai mô hình Deep Learning gồm
MobileNet V2 phục vụ cho việc nhận diện đối tượng tín hiệu đèn giao thông, mô hình
MobileNet V3 Small phục vụ cho việc xác định góc lái xe.
Chúng tôi đã thực nghiệm trên Kit Jetson Nano, xe đã hoàn thành được 7 vòng
liên tục trên bản đồ mô hình với tốc độ xử lý là 12FPS.

"""


print(clean_text(text))  
