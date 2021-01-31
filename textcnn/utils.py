import paddle
import paddle.nn.functional as F
import numpy as np
import copy

def ids_to_str(ids,word_dict):
    #print(ids)
    words = []
    for k in ids:
        w = list(word_dict)[k]
        words.append(w if isinstance(w, str) else w.decode('ASCII'))
    return " ".join(words)

def create_padded_dataset(dataset,seq_len,pad_id):
    padded_sents = []
    labels = []
    for batch_id, data in enumerate(dataset):
        sent, label = data[0], data[1]
        padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))]).astype('int32')
        padded_sents.append(padded_sent)
        labels.append(label)
    return np.array(padded_sents), np.array(labels)


