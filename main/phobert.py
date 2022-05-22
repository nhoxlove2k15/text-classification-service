from transformers import TFRobertaModel, RobertaConfig, AdamW
path_bert = "../resource/phobert/"
config = RobertaConfig.from_pretrained(
    path_bert + "config.json", from_tf=True, dropout=0.2, attention_dropout=0.2
)
print("1")

BERT_SA = TFRobertaModel.from_pretrained(
    path_bert + "model.bin",
    config=config, from_pt=True
)
print("2")
# khởi tạo, set up ma trận sẵn
import tensorflow as tf
number_labels = 37 
MAX_LEN = 256
# input_ids_in, input_masks_in, outputs dùng Bert_SA (phobert) là tham số để tạo model
input_ids_in = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_token', dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(MAX_LEN,), name='masked_token', dtype='int32') 
# print(input_ids_in, input_ids_in.shape)
outputs = BERT_SA(input_ids_in,attention_mask = input_masks_in)[0]
X= tf.keras.layers.GlobalAveragePooling1D()(outputs)
X = tf.keras.layers.Dropout(0.5)(X)

X = tf.keras.layers.Dense(number_labels, activation='sigmoid')(X)
model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = X)
print(3)
model.load_weights("../resource/my_model_7/myModelWeight.h5")
print(4)
print(model.summary())
print(5)