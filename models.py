import torch
import torch.nn as nn
import torch.nn.functional as F

def RELU(x):
    return x.relu()
def POOL(x):
    return F.max_pool2d(x,2)
def POOL1d(x):
    return F.max_pool1d(x,2)
def root(x):                                                                
    return x.sqrt()                                                     
def log(x):                                                                 
    return x.log()                                                          
def exp(x):                                                                 
    return x.exp()                                                          
def oneplus(x):                                                             
    return 1 + x                                                            
def div(x,y):                                                               
    return x/y       

class SmallNet(nn.Module):
    def __init__(self, D_in,D_out):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(D_in,128)
        self.fc2 = nn.Linear(128,D_out)
    def forward(self,x):
        h=RELU(self.fc1(x))
        return self.fc2(h)

class BigNet(nn.Module):
    def __init__(self, D_in,D_out):
        super(BigNet, self).__init__()

        self.fc1 = nn.Linear(D_in,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256,128)
        self.fc6 = nn.Linear(128,D_out)
    def forward(self,x):
        h=RELU(self.fc1(x))
        h=RELU(self.fc2(h))
        h=RELU(self.fc3(h))
        h=RELU(self.fc4(h))
        h=RELU(self.fc5(h))
        h=self.fc6(h)
        return h

class DeepNet(nn.Module):
    def __init__(self, D_in, D_out, HS, dropout_rate,bn=False):
        super(DeepNet, self).__init__()
       
        self.bn=bn

        if self.bn:
            self.bn1 = nn.BatchNorm1d(D_in)
            self.bn2 = nn.BatchNorm1d(HS[0])
            self.bn3 = nn.BatchNorm1d(HS[1])
            self.bn4 = nn.BatchNorm1d(HS[2])

        self.fc1 = nn.Linear(D_in, HS[0])
        self.fc2 = nn.Linear(HS[0], HS[1])
        self.fc3 = nn.Linear(HS[1], HS[2])
        self.fc4 = nn.Linear(HS[2], D_out)
        self._init_weights()

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def _init_weights(self):
        for name,param in self.named_parameters():
            nn.init.uniform_(param, a=-0.01,b=0.01)

    def forward(self, x):
        if self.bn:
            x = self.bn1(x)
        x = self.drop1(RELU(self.fc1(x)))
        if self.bn:
            x = self.bn2(x)
        x = self.drop2(RELU(self.fc2(x)))
        if self.bn:
            x = self.bn3(x)
        x = RELU(self.fc3(x))
        if self.bn:
            x = self.bn4(x)
        pred = self.fc4(x)
        return pred
        

class CatNN(nn.Module):
    def __init__(self, args):
        super(CatNN, self).__init__()
        if args.dataset in ['gamma']:
            hidden_sizes=[128, 64, 64]
            #hidden_sizes=[128,256,64]
        elif args.dataset in args.realsets:
            hidden_sizes=[128,256,64]
        
        self.args = args
        
        if args.kminusone==1:
            self.model = DeepNet(args.D_in,args.K-1,hidden_sizes,args.dropout_rate)
        elif args.kminusone==0:
            self.model = DeepNet(args.D_in,args.K,hidden_sizes,args.dropout_rate)
        else:    
            assert False
 
    def forward(self, src):
        return self.model(src)



class ConvHelper(nn.Module):
    def __init__(self,args):
        super(ConvHelper,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1)
        self.conv3 = nn.Conv2d(16,32,4,1)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop1 = nn.Dropout2d(args.dropout_rate)
        self.drop2 = nn.Dropout2d(args.dropout_rate)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


class CatNNConv(nn.Module):
    def __init__(self,args):
        super(CatNNConv,self).__init__()
        hidden_sizes = [512,256,64]
        self.conv = ConvHelper(args)
        self.args = args
                                                                                                                                                                                                            
        if args.kminusone==1:
            self.ff = DeepNet(32,args.K-1,hidden_sizes,args.dropout_rate,bn=True)
        elif args.kminusone==0:
            self.ff = DeepNet(32,args.K,hidden_sizes,args.dropout_rate,bn=True)
        else:
            assert False
        
    def forward(self,x):
        h = self.conv(x)
        params_t = self.ff(h)
        return params_t
