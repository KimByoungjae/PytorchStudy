import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text,self).__init__()
        self.args = args

        vocab = args.embed_num   #embedding vocab number
        dim = args.embed_dim   #300
        class_num = args.class_num   #2
        channel = 1
        k_num = args.kernel_num #100
        kernel_size = args.kernel_sizes #3,4,5

        self.embed = nn.Embedding(vocab, dim)


        self.convs = nn.ModuleList([nn.Conv2d(channel, k_num, (k, dim)) for k in kernel_size])
        """
        self.conv1 = nn.Conv2d(channel, k_num, (kernel_size[0], dim))
        self.conv2 = nn.Conv2d(channel, k_num, (kernel_size[1], dim))
        self.conv3 = nn.Conv2d(channel, k_num, (kernel_size[2], dim))
        """
        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(len(kernel_size)*k_num, class_num)

    def forward(self, x):

        x = self.embed(x) # (N,W,D)
        #print(x)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1) # (N,channel,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        """
        temp_x = x
        temp_x = F.relu(self.conv1(temp_x)).squeeze(3)  #[(N,k_num,W), ...]
        temp_x = F.max_pool1d(temp_x, x.size(2)).squeeze(2) #[(N,k_num), ...]
        x1 = temp_x

        temp_x = x
        temp_x = F.max_pool1d(F.relu(self.conv2(temp_x)).squeeze(3), x.size(2)).squeeze(2)
        x2 = temp_x

        temp_x = x
        temp_x = F.max_pool1d(F.relu(self.conv3(temp_x)).squeeze(3), x.size(2)).squeeze(2)
        x3 = temp_x
        """

        x = torch.cat(x, 1)

        x = self.dropout(x) # (N,len(kernel_size)*k_num)
        logit = self.fc1(x) # (N,class_num)
        return logit
