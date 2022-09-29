import scipy as sp
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


tf.get_logger().setLevel('ERROR')

tfhub_handle_preprocess  ="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder    = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess  )


text_test = ['this is such an amazing movie!', 'The film was nice to see.', 'I don\'t like my tea!', 'this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Shape : {text_preprocessed["input_type_ids"].shape}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12,:10]}')

s1 = bert_results["pooled_output"][0]
s2 = bert_results["pooled_output"][1]
s3 = bert_results["pooled_output"][2]
s4 = bert_results["pooled_output"][3]

print(sp.spatial.distance.cosine(s1, s2))
print(sp.spatial.distance.cosine(s2, s3))
print(sp.spatial.distance.cosine(s1, s3))
print(sp.spatial.distance.cosine(s1, s4))