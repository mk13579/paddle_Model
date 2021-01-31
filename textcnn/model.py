import paddle
import paddle.nn.functional as F
import numpy as np



class MyNet(paddle.nn.Layer):
    def __init__(self,vocab_size,emb_size):
        super(MyNet, self).__init__()
        self.emb = paddle.nn.Embedding(vocab_size, emb_size)

        self.cnn_3 = paddle.nn.Conv1D(in_channels=emb_size, out_channels=100,kernel_size=(3))
        self.cnn_4 = paddle.nn.Conv1D(in_channels=emb_size, out_channels=100,kernel_size=(4))
        self.cnn_5 = paddle.nn.Conv1D(in_channels=emb_size, out_channels=100,kernel_size=(5))

        
        self.Maxpool=paddle.nn.MaxPool1D(emb_size)

        self.fc = paddle.nn.Linear(in_features=emb_size, out_features=2)
        self.dropout = paddle.nn.Dropout(0.5)

    def forward(self, x):
        x = self.emb(x)
        x = paddle.transpose(x,[0,2,1])


        x_3=paddle.squeeze(self.Maxpool(self.cnn_3(x)),-1)
        x_4=paddle.squeeze(self.Maxpool(self.cnn_4(x)),-1)
        x_5=paddle.squeeze(self.Maxpool(self.cnn_5(x)),-1)

        x=paddle.concat([x_3,x_4,x_5],axis=1)

        x = self.dropout(x)
        x = F.softmax(self.fc(x),axis=-1)

        return x