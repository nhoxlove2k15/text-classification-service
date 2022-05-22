# -*- coding: utf-8 -*-
import clean_text
from phobert import model
from fairseq import *
from fairseq.data import Dictionary
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

# import tf.keras.utils.pad_sequences
# Load the dictionary
path_bert = "../resource/phobert/"
vocab = Dictionary()
vocab.add_from_file(path_bert + "dict.txt")
MAX_LEN = 256

def PredictNumbers(text):
# text = """
# Bảo đảm tính toàn vẹn dữ liệu bằng các dịch vụ xác thực
# Trong xã hội thông tin ngày nay, với sự phát triển của mạng Internet cùng những giao dịch điện tử như thương mại điện tử, trao đổi các nội dung số và chính phủ điện tử thì bảo mật và an toàn thông tin đang rất được quan tâm.
# Một số quốc gia trên thế giới đã và đang đầu tư mạnh mẽ cho việc nghiên cứu và xây dựng hạ tầng cơ sở cho dịch vụ an toàn trong viễn thông, bao gồm 4 dịch vụ cơ bản: Bí mật, xác thực, đảm bảo tính toàn vẹn thông tin và chống chối bỏ.
# Các dịch vụ này gắn chặt với lý thuyết mật mã và an toàn thông tin. Trong bối cảnh hội cạnh tranh và hội nhập như hiện nay, ngoài việc mua và sử dụng các công nghệ tiên tiến, chúng ta cần phải tìm hiểu và xây dựng những hướng đi riêng của mình đặc biệt là trong lĩnh vực an ninh - một lĩnh vực nhạy cảm và không được tự do hoá đáp ứng cho từng ngành khác nhau,
# Với những lý do trên, luận văn hướng tới khảo sát hệ thống an toàn thông tin cho khách hàng qua việc kiểm tra thanh toán lương tự động của khách hàng và đảm bảo an toàn qua tài khoản ATM khi rút tiền. Tiến tới mô hình lưu thông tiền tệ không dùng tiền mặt của chính phủ, đó là một trong những vấn đề trọng tâm mà ngân hàng đề ra nhằm đảm bảo an toàn dữ liệu cho khách hàng trong mọi lúc, mọi nơi. Do thời gian có hạn nên trong luận văn này em xin trình bày chương trình kiểm tra lương tự động và mô tả môđun phân hệ bảo mật máy chủ - Host Security Module (HSM) nhằm cung cấp các hàm mã hoá cần thiết để thực hiện việc mã hoá khoá, xác thực thông báo và mã hoá số PIN trong môi trường thời gian thực.
# Trên cơ sở lý thuyết an toàn dữ liệu, và kết hợp với các thuật toán cổ điển luận văn xây dựng chương trình và mô hình ứng dụng nhằm đảm bảo tính toàn vẹn của dữ liệu bằng các dịch vụ xác thực trong hệ thống Ngân hàng Đầu tư và Phát triển Việt Nam.
# Luận văn nghiên cứu và tìm hiểu các nội dung chính sau: Chương 1: Tổng quan về bảo mật và an toàn dữ liệu, Nghiên cứu, tìm hiểu các giải pháp đảm bảo an ninh an toàn mạng.  Nghiên cứu các công cụ và phương pháp về bảo mật. Một số các dịch vụ xác thực. Chương 2: Các hệ khoá mật mã và hàm băm. Hàm băm không khoá, Hàm băm có khoá. Chương 3: Các phương pháp đảm bảo tính toàn vẹn của dữ liệu.  Một số phương pháp cung cấp tính toàn vẹn của dữ liệu sử dụng hàm băm. Các phương pháp xác thực tỉnh nguyên bản của dữ liệu, Chương 4: Nghiên cứu xây dựng mô hình xác thực tính toàn vẹn của dữ liệu,

# """
  test_id = []
  text = clean_text.clean_text(text)
  # subwords = '<s> ' + bpe.encode(text) + ' </s>'
  subwords =  text

  encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
  test_id.append(encoded_sent)

  test_id = pad_sequences(test_id, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

  test_mask = []
  print(test_id)
  mask = [int(token_id > 0) for token_id in test_id[0]]
  test_mask.append(mask)
  predicts = model.predict([test_id,np.array(test_mask)])
  return predicts
  # print(predicts)

# predict ==================================================================
# print(text)
def classify(predicts,threshhold = 0.1):
  preds = []
  for idx,predict in enumerate(predicts):
    a = []
    preds.append(int(idx+1))
    for i in range (0, len(predict)):
        if predict[i] >= threshhold:
          a.append(classes[i])
    preds.append(a)
  return preds
def top_n(predict, n = 5):
  output = sorted(range(len(predict)), key=lambda k: predict[k], reverse=True)
  # output = predict.sort(reverse=True)
  return output[:n]
# vector hóa tags

classes = [
           'mạng neural','phân loại','xử lý ảnh','di động','web','ids','gan','sdn','học máy','khuyến nghị','bảo mật','ai', 
           'raspberry pi','jetson nano','devops','mạng 5g','blockchain','camera','tự động hóa', 'học sâu', 'lập trình nhúng', 'fpga',
           'mạng bluetooth', 'cloud','tự hành', 'mạng wifi', 'rfid' ,'yolo','robot','nhận diện', 'iot','vr','e-commerce','game','ar','nlp','mạng xã hội'
]
# [0 0 1 0 ...0 1 ...]

number_labels = len(classes)
def to_category_vector(label):
    # print(label[-1])
    # print(label)
    vector = np.zeros(len(classes)).astype(np.float64)
    # print(vector)
    if label[-1] == ',':
      # a ='KH,SK,DL,SK,'
      # b = a.split(',') # ["KH","SK","DL","SK",""]
      a = label.split(',')[:-1]
      # a = ['xử lý ảnh', 'web']
      for i in a :
        index = classes.index(i.strip())
        # print(index)
        vector[index] = 1.0
      # return
    else:
    

      index = classes.index(label)
      # print(index)
      vector[index] = 1.0
      # vector[2] = 1.0
    return vector
def PredictLabels(predicts):    
  y_preds = []
  y_preds_string = []  
  for idx,y in enumerate(predicts):
    top_5 = top_n(predicts[idx].tolist(), 5)
    tags_5 = [classes[i] for i in top_5]
    string = ', '.join([str(item) for item in tags_5])
    y_preds_string.append(string+",")
    if idx == 0 : print(string+",")
    # print(string[-1])
    y_preds.append(to_category_vector(string+","))
  return y_preds_string
# top_n(predicts[0].tolist(), 4)
# print(y_preds_string)
