import paddle
import paddle.nn.functional as F
import numpy as np
import copy
from model import *
from utils import *
from data import *

print(paddle.__version__)



print('loading dataset...')
train_dataset = paddle.text.datasets.Imdb(mode='train')
test_dataset = paddle.text.datasets.Imdb(mode='test')
print('loading finished')


word_dict = train_dataset.word_idx

# add a pad token to the dict for later padding the sequence
word_dict['<pad>'] = len(word_dict)

for k in list(word_dict)[:5]:
    print("{}:{}".format(k.decode('ASCII'), word_dict[k]))

print("...")

for k in list(word_dict)[-5:]:
    print("{}:{}".format(k if isinstance(k, str) else k.decode('ASCII'), word_dict[k]))

print("totally {} words".format(len(word_dict)))


vocab_size = len(word_dict) + 1
emb_size = 300
seq_len = 50
batch_size = 32
epochs = 10
pad_id = word_dict['<pad>']

classes = ['negative', 'positive']



sent = train_dataset.docs[0]
label = train_dataset.labels[1]
print('sentence list id is:', sent)
print('sentence label id is:', label)
print('--------------------------')
print('sentence list is: ', ids_to_str(sent,word_dict))
print('sentence label is: ', classes[label])



train_sents, train_labels = create_padded_dataset(train_dataset,seq_len,pad_id)
test_sents, test_labels = create_padded_dataset(test_dataset,seq_len,pad_id)

print(train_sents.shape)
print(train_labels.shape)

print(test_sents.shape)
print(test_labels.shape)

for sent in train_sents[:3]:
    print(ids_to_str(sent,word_dict))




train_dataset = IMDBDataset(train_sents, train_labels)
test_dataset = IMDBDataset(test_sents, test_labels)

train_loader = paddle.io.DataLoader(train_dataset
                                    ,return_list=True
                                    , shuffle=True
                                    ,batch_size=batch_size
                                    ,drop_last=True
                                    )                                
test_loader = paddle.io.DataLoader(test_dataset
                                    ,return_list=True
                                    ,shuffle=True
                                    ,batch_size=batch_size
                                    ,drop_last=True
                                    )



model=MyNet(vocab_size,emb_size)

optim = paddle.optimizer.Adam(learning_rate=0.001
                             ,parameters=model.parameters()
                             )

Loss=paddle.nn.CrossEntropyLoss()
# 模型配置




def train_or_eval_or_preict(model=model
                           ,dataloader=None
                           ,optim=None
                           ,Loss=None
                           ,epochs=None
                           ,mode=None
                           ):
    if mode=="train":
        model.train()
    else:
        model.eval()
    
    Predict = []
    Label = []
    Loss_list=[]
    batch_max=0
    for batch_id,data in enumerate(dataloader):
        batch_max=batch_id+1

        sent = data[0]
        label = data[1]
        logits = model(sent)

        loss = Loss(logits, label)

        if mode == "train":
            loss.backward()
            optim.step()
            optim.clear_grad()
        
        ans=paddle.argmax(logits,axis=1).numpy().tolist()
        label=label.squeeze().numpy().tolist()
        loss=loss.numpy().tolist()

        Predict.extend(ans)
        Label.extend(label)
        Loss_list.extend(loss)
    
    Predict=np.array(Predict)
    Label=np.array(Label)
    Loss_list=np.array(Loss_list)




    total=len(Label)
    Loss=np.sum(Loss_list)/batch_max
    acc=np.sum(Predict==Label)
    acc=100*acc/total

    
    return np.round(acc,2),np.round(Loss,4)


        
            
              
best_acc=0.0
best_model=None
for i in range(epochs):
    acc,loss=train_or_eval_or_preict(model=model,dataloader=train_loader,optim=optim,Loss=Loss,epochs=epochs,mode="train")
    print("train ---- epoch:[%d/%d]  acc: %.2f ,loss: %.4f "%(i+1,epochs,acc,loss))

    acc,loss=train_or_eval_or_preict(model=model,dataloader=test_loader,optim=optim,Loss=Loss,epochs=epochs,mode="dev")
    print("dev ---- epoch:[%d/%d]  acc: %.2f ,loss: %.4f "%(i+1,epochs,acc,loss))

    if best_model==None or best_acc < acc :
       best_acc=acc
       best_model=copy.copy(model)


acc,_=train_or_eval_or_preict(model=best_model,dataloader=test_loader,optim=optim,Loss=Loss,epochs=epochs,mode="test")

print("final acc : %.2f"%acc)


