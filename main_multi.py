import scipy.stats as stats
from scipy.special import expit
from random import seed
from random import randrange
import pandas as pd
import scipy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle
from utils_crm import *
from data_simulation import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
import argparse
import random
from skmultilearn.dataset import load_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
import time
import sys

from matplotlib import pyplot as plt
import torch
import pandas as pd
import os
term_size = os.get_terminal_size()

parser = argparse.ArgumentParser()
parser.add_argument("--ifdef", help = "deficient logging policy", default = 0)
parser.add_argument("--nrep", help = "number of simulations", default = 5)
parser.add_argument("--dataset", help = "price settings", type=str,default = 'yeast')
parser.add_argument("--n", help = "number of samples", default = 2500)
parser.add_argument("--biased", help = "biased logging policy", default = 0)
parser.add_argument("--prob_reg", help = "prob regularizer", default = 0)
parser.add_argument('--use_cv', default = 0)
parser.add_argument('--debiased_cv', default = 0)
parser.add_argument('--dropout', default = 1)
parser.add_argument('--test_ratio', default = 0.8)
parser.add_argument('--datasetname', default = 1)
parser.add_argument('--hidd', default = 128)
parser.add_argument('--niterations', default = 2)
parser.add_argument('--backbone', default = 'dm')
parser.add_argument('--stmodel', default = 'st')
parser.add_argument('--hsiclamb', default = 0.01)
parser.add_argument('--ovp', default = 0)

args = parser.parse_args()
if_def = int(args.ifdef)
nrep = int(args.nrep)
dataset = args.dataset
nsamples = int(args.n)
biased = int(args.biased)
prob_reg = float(args.prob_reg)
use_cv = int(args.use_cv)
debiased_cv = int(args.debiased_cv)
dropout = int(args.dropout)
test_ratio = float(args.test_ratio)
datasetname = int(args.datasetname)
hidd = int(args.hidd)
niterations = int(args.niterations)
backbone = str(args.backbone)
stmodel = str(args.stmodel)
hsiclamb = float(args.hsiclamb)
ovp = float(args.ovp)

class Net(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden = 8):
        super(Net, self).__init__()
        self.emb_layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden)
            )
        self.outlayers = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),            
            torch.nn.Linear(hidden, num_classes),
            torch.nn.Dropout(p=0.2),
            )
        self.proplayers = torch.nn.Sequential(
            torch.nn.Linear(hidden, num_classes)
            )

    def emb(self, x):
        return self.emb_layers(x)

    def prop(self, x):
        emb = self.emb_layers(x)
        return self.proplayers(emb)

    def forward(self, x):
        emb = self.emb_layers(x)
        return self.outlayers(emb)


@ torch.no_grad()
def test_multi(model, X_test, y_test):
    model.eval()
    model.cpu()
    out1 = model(X_test.float())
    out = torch.sigmoid(out1)
    y_test = torch.from_numpy(y_test).float()
    _, treat = torch.max(out, 1)
    labels = y_test[range(y_test.shape[0]),treat.numpy()]
    total_reward = labels.reshape(-1).sum()
    pred = torch.where(out<=0.5, torch.zeros_like(out), torch.ones_like(out))
    ham_loss = hamming_loss(y_test.numpy(), pred.detach().numpy())
    auc = roc_auc_score(y_test.numpy(), out.detach().numpy(), average = 'micro')
    test_loss = torch.nn.MultiLabelSoftMarginLoss()(out1, y_test).detach().numpy()*1.
    print('propotions of ones gt: ', y_test.mean())
    print('propotions of ones predicted: ', pred.mean())
    print('test loss: ', torch.nn.MultiLabelSoftMarginLoss()(out1, y_test))
    print('hamming loss:', hamming_loss(y_test.numpy(), pred.detach().numpy()))
    print('total reward: ', total_reward)
    print('auc: ', auc)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    return total_reward, ham_loss, test_loss, auc


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def train_dm(trainloader, input_dim, output_dim, lamb = 0, num_epochs = 1000):
    model = Net(input_dim, output_dim, hidden = hidd)
    criterion = torch.nn.BCEWithLogitsLoss()
    cmcriterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1, momentum = 0.9, weight_decay = 1e-6)
    min_loss = 1e3
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputx, treatment, targety, logging_prob) in enumerate(trainloader):
            # forward
            out = model(inputx.float())
            out = out[range(out.size(0)),treatment]
            loss = criterion(out.reshape(-1), targety.float().reshape(-1))
            # backward
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, running_loss / (batch_idx+1)))
    return model

def select_lambda_hsic(dataset, input_dim, output_dim, lamb = [1], nfolds=3):
    N = X.shape[0]  # total num of samples
    n = N // nfolds  # num of samples per fold
    # shuffle
    shuf_index = torch.randperm(N).numpy()
    losses = [0] * len(lamb)
    for j in range(len(lamb)):
        this_lamb = lamb[j]
        loss = 0
        for i in range(nfolds):
            # get subsets
            idx_full = range(N)
            idx_test = range(i*n, (i+1)*n)
            idx_train = list(set(idx_full)-set(idx_test))
            train_set = torch.utils.data.dataset.Subset(dataset,idx_train)
            val_set = torch.utils.data.dataset.Subset(dataset,idx_test)
            trainloader = DataLoader(train_set, batch_size = 128)
            valloader = DataLoader(val_set, batch_size = 128)
            # train
            model = train_hsic(trainloader, input_dim, output_dim, this_lamb)
            # test
            this_loss = test_hsic_cv(valloader, model)
            loss += this_loss
        losses[j] = loss / nfolds
    print(losses)
    best_lamb = lamb[losses.index(min(losses))]
    return best_lamb

def train_HSIC_CV(dataset, dataloader, input_dim, output_dim, lamb=[1], nfolds=3):
    best_lamb = select_lambda_hsic(dataset, input_dim, output_dim, lamb, nfolds)
    print(f'=> the best lambda is {best_lamb}')
    best_model = train_hsic(dataloader, input_dim, output_dim, best_lamb)
    return best_model

@ torch.no_grad()
def test_hsic_cv(loader, model):
    total_loss = 0
    for batch_idx, (bat_X, bat_treatment, bat_y, logp) in enumerate(loader):
        out = model(bat_X.float())
        out = out[range(out.size(0)), bat_treatment]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, bat_y.float())
        total_loss += loss.item()
    return total_loss

def train_hsic(trainloader, input_dim, output_dim, lamb = 0, num_epochs = 2000):
    model = Net(input_dim, output_dim, hidden = hidd)
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.BCEWithLogitsLoss()
    cmcriterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    kernel = RBFkernel()
    min_loss = 1e3
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (bat_X, bat_treatment, bat_y, logp) in enumerate(trainloader):
            out = model(bat_X.float())
            emb = model.emb(bat_X.float())
            bat_treatoh = torch.eye(torch.unique(treatment).shape[0])[bat_treatment]
            if torch.cuda.is_available():
                bat_treatoh.cuda()
            if lamb > 0 :
                est_hsic = hsic(kernel, kernel, bat_X.shape[0], emb, bat_treatoh)
                out = out[range(out.size(0)),bat_treatment]
                loss = criterion(out, bat_y.float().reshape(-1)) + lamb * est_hsic.mean()
            else:
                loss = criterion(out, bat_y.long()).mean() #+ 10 * est_hsic
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, running_loss / (batch_idx+1)))
    return model



def select_lambda_ips(input_dim, output_dim, X, y, treatment, log_model, lamb=[1], nfolds=3):
    find_max = True
    N = X.shape[0]  # total num of samples
    n = N // nfolds  # num of samples per fold
    # shuffle
    shuf_index = torch.randperm(N).numpy()
    X = X[shuf_index]
    y = y[shuf_index]
    treatment = treatment[shuf_index]
    losses = [0] * len(lamb)
    for j in range(len(lamb)):
        print('fold ',j,':')
        this_lamb = lamb[j]
        loss = 0
        for i in range(nfolds):
            # get subsets
            idx_full = range(N)
            idx_test = range(i*n, (i+1)*n)
            idx_train = list(set(idx_full)-set(idx_test))
            X_test = X[idx_test]
            y_test = y[idx_test]
            treatment_test = treatment[idx_test]
            X_train = X[idx_train]
            y_train = y[idx_train]
            treatment_train = treatment[idx_train]
            # train
            model = train_ips(input_dim, output_dim, X_train, y_train, treatment_train, log_model, this_lamb)
            # test
            this_loss = test_ips_cv(model, X_test, y_test, treatment_test)
            loss += this_loss
        losses[j] = loss / nfolds
    print(losses)
    losses = [-l if find_max else l for l in losses]
    best_lamb = lamb[losses.index(min(losses))]  # select max prob
    return best_lamb


@ torch.no_grad()
def test_ips_cv(model, X, y, treatment):
    if torch.cuda.is_available():
        inputX = Variable(torch.from_numpy(X)).cuda()
        targety = Variable(y).cuda()
        ttreatment = Variable(torch.from_numpy(treatment)).cuda()
        model.cuda()
    else:
        inputX = Variable(torch.from_numpy(X))
        targety = Variable(y)
        ttreatment = Variable(torch.from_numpy(treatment))
    out = model(inputX.float())
    # out = F.softmax(out, dim = 1)
    loss = torch.nn.functional.cross_entropy(out, ttreatment.long(), reduction='none')
    prob = torch.exp(-loss) * targety
    prob_mean = prob.mean()
    return prob_mean

@ torch.no_grad()
def test_dm_cv(model, X, y, treatment):
    if torch.cuda.is_available():
        inputX = Variable(torch.from_numpy(X)).cuda()
        targety = Variable(y).cuda()
        ttreatment = Variable(torch.from_numpy(treatment)).cuda()
        model.cuda()
    else:
        inputX = Variable(torch.from_numpy(X))
        #inputx = Variable(x)
        targety = Variable(y)
        ttreatment = Variable(torch.from_numpy(treatment))
    out = model(inputX.float())
    loss = torch.nn.functional.cross_entropy(out, ttreatment.long(), reduction='none')
    loss = loss.mean()
    return loss

@ torch.no_grad()
def test_dm_cv_rand(model, X, y, treatment, log_model):
    if torch.cuda.is_available():
        inputX = Variable(torch.from_numpy(X)).cuda()
        #inputx = Variable(x).cuda()
        targety = Variable(y).cuda()
        ttreatment = Variable(torch.from_numpy(treatment)).cuda()
        model.cuda()
    else:
        logp = log_model.predict_proba(X)[:,treatment]
        logp = torch.from_numpy(logp)
        inputX = torch.from_numpy(X)
        #inputx = Variable(x)
        targety = y
        ttreatment = torch.from_numpy(treatment)
    out = model(inputX.float())
    # out = F.softmax(out, dim = 1)
    loss = torch.nn.functional.cross_entropy(out, ttreatment.long(), reduction='none') / logp
    loss = loss.mean()
    #prob = torch.exp(-loss) * targety
    #prob_mean = prob.mean()
    return loss

def train_IPS_CV(input_dim, output_dim, X, y, treatment, log_model, lamb=[1], nfolds=3):
    y = y.reshape(-1)
    treatment = treatment.reshape(-1)
    best_lamb = select_lambda_ips(input_dim, output_dim, X, y, treatment, log_model, lamb, nfolds)
    print(f'=> the best lambda is {best_lamb}')
    best_model = train_ips(input_dim, output_dim, X, y, treatment, log_model, best_lamb)
    return best_model

def train_ips(trainloader, input_dim, output_dim, lamb = 0):
    model = Net(input_dim, output_dim, hidden = hidd)
    criterion = torch.nn.BCEWithLogitsLoss()
    cmcriterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1, momentum = 0.9, weight_decay = 1e-6)
    num_epochs = 10000
    min_loss = 1e3
    min_loss = 1e3
    running_loss =0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputx, treatment, targety, logging_prob) in enumerate(trainloader):
            # forward
            out = model(inputx.float())
            out = F.softmax(out, 1)
            out = out[range(out.size(0)),treatment]
            logp = logging_prob[range(out.size(0)),treatment]
            reward = targety
            loss = - (reward - lamb) * out / logp
            loss = loss.mean()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 1000 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, running_loss / (batch_idx+1)))
    return model

def train_wang(trainloader, input_dim, output_dim, num_epochs = 2000):
    model = Net(input_dim, output_dim)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    cmcriterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    running_loss =0
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputx, treatment, targety, logging_prob) in enumerate(trainloader):
        # forward
            out = model(inputx.float())
            out = out[range(out.size(0)),treatment]
            logp = logging_prob[range(out.size(0)),treatment]
            reward = targety
            loss = criterion(out.reshape(-1), targety.float().reshape(-1)) / logp
            loss = loss.mean()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 1000 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, running_loss / (batch_idx+1)))
    return model


def train_selfdm(trainloader, input_dim, output_dim, pseudo_labeler = None, sample_imp = 1, prob_reg = 0, lambc = 0.5, model = None, num_epochs = 2000, alpha = 1):
    if model:
        import copy
        model = copy.deepcopy(model)
    else:
        model = Net(input_dim, output_dim)

    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=3)
    labeling_iter = niterations
    forward_passes = 1
    for labeling_it in range(labeling_iter):
        model = Net(input_dim, output_dim)
        criterion = torch.nn.BCEWithLogitsLoss(reduction = 'sum')
        cmcriterion = torch.nn.MultiLabelSoftMarginLoss()
        lr = 1e-3
        factual_optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        cf_optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        if torch.cuda.is_available():
            model.cuda()        
        if torch.cuda.is_available():
            pseudo_labeler.cuda()
        best_model_this_epoch = None
        best_loss = 1e5
        print(f'alpha is {alpha}')
        for epoch in range(num_epochs):
            running_loss = 0.
            stop_epochs = 1000
            num_epochs = 2000
            thres = 0.95
            quantile = 0.95
            for batch_idx, (inputx, treatment, targety, logging_prob) in enumerate(trainloader):
                # train source
                out = model(inputx.float())
                # factual loss 
                loss1 = criterion(out[range(out.size(0)),treatment.long()].reshape(-1), targety.float().reshape(-1))
                out1 = model(inputx.float())
                out = torch.sigmoid(out1).detach()

                dropout_predictions = np.empty((0, inputx.shape[0], output_dim))
                softmax = torch.nn.Softmax(dim=1)
                for i in range(forward_passes):
                    predictions = np.empty((0, output_dim))
                    pseudo_labeler.eval()
                    enable_dropout(pseudo_labeler)
                    with torch.no_grad():
                        output = pseudo_labeler(inputx.float())
                        output = torch.sigmoid(output)
                    predictions = np.vstack((predictions, output.cpu().numpy()))

                    dropout_predictions = np.vstack((dropout_predictions,
                                                     predictions[np.newaxis, :, :]))
                
                # Calculating mean across multiple MCD forward passes 
                conf = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
                # Calculating variance across multiple MCD forward passes 
                variance = np.std(dropout_predictions, axis=0) # shape (n_samples, n_classes)
                hp = 50
                lp = 50 
                high = np.percentile(conf, 100-hp, 0)#100-thres/2 * 100, 0)
                low = np.percentile(conf, lp, 0)
                varthres = np.percentile(variance, 5, 0)
                pvarthres = 2#0.05
                nvarthres = 2#0.005   
                if torch.cuda.is_available():
                    conf = torch.Tensor(conf).cuda()
                    variance = torch.Tensor(variance).cuda()
                    high = torch.Tensor(high).cuda()
                    low = torch.Tensor(low).cuda()
                    varthres = torch.Tensor(varthres).cuda()
                else:
                    conf = torch.Tensor(conf)
                    variance = torch.Tensor(variance)
                    high = torch.Tensor(high)
                    low = torch.Tensor(low)    
                    varthres = torch.Tensor(varthres)                
                pseudo_labeler.train()
                pseudo_label = torch.where(conf <= lambc, torch.zeros_like(conf), torch.ones_like(conf))
                pseudo_label[range(pseudo_label.size(0)),treatment] = targety.float().reshape(-1)

                mask = torch.ones_like(pseudo_label).scatter_(1, treatment.long().unsqueeze(1), 0)
                pmask = torch.logical_and(conf > high, variance < pvarthres) 
                nmask = torch.logical_and(conf < low, variance < nvarthres)

                pmask = torch.logical_and(mask, pmask)
                nmask = torch.logical_and(mask, nmask)
                mask = torch.logical_or(pmask, nmask)

                cf_labels = pseudo_label[mask.bool()].reshape(-1)
                conf = conf[mask.bool()].reshape(-1)
                out = out[mask.bool()].reshape(-1)
                out1 = out1[mask.bool()].reshape(-1)
                variance = variance[mask.bool()].reshape(-1)


                loss = criterion(out1, cf_labels.float()) + loss1
                vatloss = vat_loss(model, inputx.float(), treatment)
                loss /=  (inputx.shape[0] + conf.shape[0])#(inputx.shape[0]) * output_dim
                loss += alpha * vatloss
                # backward
                cf_optimizer.zero_grad()
                loss.mean().backward()
                #print(model.linear1.weight.grad)
                cf_optimizer.step()
                running_loss += loss.data.item()

            if (epoch+1) % 500 == 0:
                print('Epoch[{}/{}], loss: {:.6f}'.format(epoch, num_epochs, running_loss / (batch_idx + 1)))
                test_multi(model, X_test, y_test)
            current_loss = running_loss / (batch_idx + 1)
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_this_epoch = copy.deepcopy(model)
        pseudo_labeler = copy.deepcopy(best_model_this_epoch)
        print('=' * term_size.columns)
        print(f'labeling round {labeling_it} Finished.')
        test_multi(best_model_this_epoch, X_test, y_test)
    return best_model_this_epoch



def select_lambda_st(input_dim, output_dim, X, y, treatment, log_model, lamb=[1], nfolds=3):
    find_max = False
    N = X.shape[0]  # total num of samples
    n = N // nfolds  # num of samples per fold
    # shuffle
    shuf_index = torch.randperm(N).numpy()
    X = X[shuf_index]
    y = y[shuf_index]
    treatment = treatment[shuf_index]
    losses = [0] * len(lamb)
    for j in range(len(lamb)):
        print('fold ',j,':')
        this_lamb = lamb[j]
        loss = 0
        for i in range(nfolds):
            # get subsets
            idx_full = range(N)
            idx_test = range(i*n, (i+1)*n)
            idx_train = list(set(idx_full)-set(idx_test))
            X_test = X[idx_test]
            y_test = y[idx_test]
            treatment_test = treatment[idx_test]
            X_train = X[idx_train]
            y_train = y[idx_train]
            treatment_train = treatment[idx_train]
            model = train_selfdm(input_dim, output_dim, X_train, y_train, treatment_train, sample_imp = 0, prob_reg = 0, lamb = this_lamb)
            # test
            if not debiased_cv:
                this_loss = test_dm_cv(model, X_test, y_test, treatment_test)
            else:
                this_loss = test_dm_cv_rand(model, X_test, y_test, treatment_test, log_model)
            loss += this_loss
        losses[j] = loss / nfolds
    print(losses)
    losses = [-l if find_max else l for l in losses]
    best_lamb = lamb[losses.index(min(losses))]  # select max prob
    return best_lamb


def train_st_CV(input_dim, output_dim, X, y, treatment, log_model, lamb=[1], nfolds=3):
    y = y.reshape(-1)
    treatment = treatment.reshape(-1)
    best_lamb = select_lambda_st(input_dim, output_dim, X, y, treatment, log_model, lamb, nfolds)
    print(f'=> the best lambda is {best_lamb}')
    best_model = train_selfdm(input_dim, output_dim, X, y, treatment, sample_imp = 0, prob_reg = 0, lamb = best_lamb)
    return best_model

def set_seed(seed):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


import contextlib
import torch.nn as nn

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

#'''
class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, treatment):
        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                pred_hat = torch.sigmoid(pred_hat)
                #logp_hat = F.log_softmax(pred_hat, dim=1)
                #adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                #adv_distance = (pred_hat - pred) ** 2
                #adv_distance = (1-pred_hat) * torch.log((1-pred_hat)/(1-pred+1e-7)+1e-7) + (pred_hat) * torch.log((pred_hat)/(pred+1e-7)+1e-7)
                adv_distance = (1-pred) * torch.log((1-pred)/(1-pred_hat+1e-7)+1e-7) + (pred) * torch.log((pred)/(pred_hat+1e-7)+1e-7)
                mask = torch.ones_like(pred_hat).scatter_(1, treatment.long().unsqueeze(1), 0)
                #mask = torch.ones_like(pred_hat)
                adv_distance = adv_distance[mask.bool()].reshape(-1).mean()

                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            pred_hat = torch.sigmoid(pred_hat)
            #lds = (pred_hat - pred) ** 2
            #lds = (1-pred_hat) * torch.log((1-pred_hat)/(1-pred+1e-7)+1e-7) + (pred_hat) * torch.log((pred_hat)/(pred+1e-7)+1e-7)
            lds = (1-pred) * torch.log((1-pred)/(1-pred_hat+1e-7)+1e-7) + (pred) * torch.log((pred)/(pred_hat+1e-7)+1e-7)
            lds = lds[mask.bool()].mean()
        return lds


if use_cv:
    print('USE CV.')
    train_HSIC = train_HSIC_CV
    train_IPS = train_IPS_CV




dm_result = []
dm_loss = []
dm_softloss = []
dm_time = []
dm_auc = []
vat_result = []
vat_loss = []
vat_softloss = []
vat_time = []
vat_auc = []
wang_result = []
wang_loss = []
wang_softloss = []
wang_time = []
wang_auc = []
opt_result = []
crmdm_result = []
crmdm_mon_result = []
dmaug_result = []
dmself_result = []
dmself_loss = []
dmself_softloss = []
dmself_time = []
dmself_auc = []
dmselfarg_result = []
dmselfarg_loss = []
dmselfarg_softloss = []
dmselfarg_time = []
dmselfarg_auc = []
dmhsic_result = []
dmhsic_loss = []
dmhsic_softloss = []
dmhsic_time = []
dmhsic_auc = []
regself_result = []
kdd_all_result = []
kdd_par_result = []
ips_result = []
ips_loss = []
ips_time = []
ips_auc = []
ipsaug_result = []

name = dataset

from sklearn.preprocessing import StandardScaler

price_grid = [1,2,3,4,5]

for r in range(nrep):
    #model = generate_logging_policy(X, y, frac = 0.05)
    set_seed(81 * r + 1)
    valid_size = 500
    if dataset == 'scene':
        X, y, feature_names, label_names = load_dataset('scene', 'train')
        X_test, y_test, _, _ = load_dataset('scene', 'test')
        y = y.todense()
        X = X.todense()

        X, y, feature_names, label_names = load_dataset('scene', 'train')
        X_test, y_test, _, _ = load_dataset('scene', 'test')
        y = np.array(y.todense())
        X = np.array(X.todense())

        y_test = np.array(y_test.todense())
        X_test = np.array(X_test.todense())

        X = np.append(X, X_test, axis = 0)
        y = np.append(y, y_test, axis = 0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_full = X
        y_full = y
        #y = 2 * (y - 0.5)
        X, X_test, y, y_test, indices_train, indices_test = train_test_split(X, y, range(X.shape[0]), test_size=test_ratio, random_state = 81*r + 1)
    if dataset == 'yeast':
        X, y, feature_names, label_names = load_dataset('yeast', 'train')
        X_test, y_test, _, _ = load_dataset('yeast', 'test')
        y = y.todense()
        X = X.todense()
        
        y_test = np.array(y_test.todense())
        X_test = np.array(X_test.todense())

        X = np.append(X, X_test, axis = 0)
        y = np.append(y, y_test, axis = 0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_full = X
        y_full = y
        #y = 2 * (y - 0.5)
        X, X_test, y, y_test, indices_train, indices_test = train_test_split(X, y, range(X.shape[0]), test_size=test_ratio, random_state = 81*r + 1)        
    if dataset == 'syn':
        nsamples = 1000
        ntest = 10000
        nvalid = 500
        if_def = 0
        biased = 0
        X = sample_x(nsamples, 50)
        X_valid = sample_x(nvalid, 50)        
        X_test = sample_x(ntest, 50)
        a, b, c = sample_a(50, 50)
        hx = [h(x, a, b, c) for x in X]
        treatment = sample_treatment(X, if_def, biased, price_grid = price_grid)
        y = sample_ally(X, hx, datasetname, price_grid = price_grid)
        treatment = treatment - 1
        #features = np.append(X,treatment.reshape(-1,1),1)
        hxtest = [h(x, a, b, c) for x in X_test]
        hxval = [h(x, a, b, c) for x in X_valid]
        y_test = sample_ally(X_test, hxtest, datasetname, price_grid = price_grid)
        y_valid = sample_ally(X_valid, hxval, datasetname, price_grid = price_grid)
    if dataset == 'synovp':
        nsamples = 1000
        ntest = 10000
        nvalid = 500
        if_def = 0
        biased = 0
        X = sample_x(nsamples, 50)
        X_valid = sample_x(nvalid, 50)        
        X_test = sample_x(ntest, 50)
        a, b, c = sample_a(50, 50)
        hx = [h(x, a, b, c) for x in X]
        treatment = sample_treatment(X, if_def, biased, price_grid = price_grid, ovp = ovp)
        y = sample_ally(X, hx, datasetname, price_grid = price_grid)
        treatment = treatment - 1
        #features = np.append(X,treatment.reshape(-1,1),1)
        hxtest = [h(x, a, b, c) for x in X_test]
        hxval = [h(x, a, b, c) for x in X_valid]
        y_test = sample_ally(X_test, hxtest, datasetname, price_grid = price_grid)
        y_valid = sample_ally(X_valid, hxval, datasetname, price_grid = price_grid)        

    print('Current Rep: ', r)
    print('Dataset : ', dataset)
    print('Deficient Logging Policy: ', if_def)
    print('Number of samples: ', X.shape[0])

    print(y)
    print(y.shape)
    y_all = y

    X_valid = X_test[:valid_size]
    y_valid = y_test[:valid_size]
    X_test = X_test[valid_size:]
    y_test = y_test[valid_size:]

    if dataset == 'syn':
        X_valid = torch.from_numpy(X_valid) 
        y_valid = y_valid
        X_test = torch.from_numpy(X_test)
        y_test = y_test
    else:
        X_valid = torch.from_numpy(X_valid) 
        y_valid = y_valid
        X_test = torch.from_numpy(X_test)
        y_test = y_test

    set_seed(81*r + 1)
    log_size = X.shape[0] // 5
    log_index = np.random.choice(range(X.shape[0]),log_size,replace=False)
    print(log_index)
    alt_human = MultiOutputClassifier(RandomForestClassifier()).fit(X[log_index], y[log_index])

    from scipy.special import softmax
    # randomized selection
    #treatment = np.argmax(np.stack(alt_human.predict_proba(X))[:,:,1].T,axis=1)
    problog = np.stack(alt_human.predict_proba(X))[:,:,1].T
    temp = 3
    problog = softmax(problog * temp, axis = 1)
    print(problog)
    if 'syn' not in dataset:
        treatment = np.array([np.random.choice(range(y_all.shape[1]),p=problog[i]) for i in range(problog.shape[0])])

    print(treatment)
    print(set(treatment))
    y = y_all[range(y_all.shape[0]), treatment]
    y = torch.from_numpy(y)


    if args.use_cv:
        train_HSIC = train_HSIC_CV
        train_ips = train_IPS_CV


    log_est = LogisticRegression(random_state=0, solver = 'liblinear', max_iter = 1000).fit(X, treatment)

    logging_prob = log_est.predict_proba(X)
    print('logging:')
    print(logging_prob.shape)
    logging_prob = torch.from_numpy(logging_prob)

    X = torch.from_numpy(X)
    treatment = torch.from_numpy(treatment)

    if torch.cuda.is_available():
        data = TensorDataset(X.cuda(), treatment.cuda(), y.reshape(-1,1).cuda(), logging_prob.cuda())
    else:
        data = TensorDataset(X, treatment, y.reshape(-1,1), logging_prob)
    trainloader = DataLoader(data, batch_size = 64)

    if backbone == 'dm':    
        set_seed(r)
        start = time.time()
        dm = train_dm(trainloader, X.shape[1],y_all.shape[1], num_epochs = 2000)
        end = time.time()
        rev, hamm, tl, auc = test_multi(dm, X_test, y_test)
        print('DM Rev = %.2f' % rev)
        dm_result.append(rev)
        dm_loss.append(hamm)
        dm_softloss.append(tl)
        dm_time.append(end-start)
        dm_auc.append(auc)
        dmval_rev, dmval_hamm, dmval_tl, dmval_auc = test_multi(dm, X_valid, y_valid)
    elif backbone == 'hsic':
        set_seed(r)
        start = time.time()
        if use_cv:
            dm = train_HSIC(data, trainloader, X.shape[1],y_all.shape[1], lamb = [1e-2,1e-1,1,1e1,1e2])
        else:
            dm = train_hsic(trainloader, X.shape[1],y_all.shape[1], lamb = hsiclamb, num_epochs = 1000)
        end = time.time()
        rev, hamm, tl, auc = test_multi(dm, X_test, y_test)
        print('DM with HSIC Rev = %.2f' % rev)
        dmhsic_result.append(rev)
        dmhsic_loss.append(hamm)
        dmhsic_softloss.append(tl)
        dmhsic_time.append(end-start)
        dmhsic_auc.append(auc)
        dmval_rev, dmval_hamm, dmval_tl, dmval_auc = test_multi(dm, X_valid, y_valid)        
    elif  backbone == 'wang':
        set_seed(r)
        start = time.time()
        dm = train_wang(trainloader, X.shape[1],y_all.shape[1], num_epochs = 3000)
        end = time.time()
        rev, hamm, tl, auc = test_multi(dm, X_test, y_test)
        print('Wang = %.2f' % rev)
        wang_result.append(rev)
        wang_loss.append(hamm)
        wang_softloss.append(tl)
        wang_time.append(end-start)
        wang_auc.append(auc)
        dmval_rev, dmval_hamm, dmval_tl, dmval_auc = test_multi(dm, X_valid, y_valid)

    print('=' * term_size.columns)
    if stmodel == 'st' or stmodel == 'both':
        print('Starting ST:')
        import copy
        dm1 = copy.deepcopy(dm)
        pseudo_labeler = copy.deepcopy(dm)
        dmself = train_selfdm(trainloader, X.shape[1],y_all.shape[1], pseudo_labeler = pseudo_labeler, sample_imp = 0, prob_reg = 0, model = dm1, num_epochs = 3000, lambc = 0.5, alpha = 0)
        rev, hamm, tl, auc = test_multi(dmself, X_test, y_test)
        dmself_result.append(rev)
        dmself_loss.append(hamm)
        dmself_softloss.append(tl)
        dmself_time.append(end-start)
        dmself_auc.append(auc)

    print('=' * term_size.columns)
    if stmodel == 'stvat' or stmodel == 'both':
        print('Starting ST + VAT :')
        dmself_alpha_list = []
        alpha_perf = []
        alphas = [1e-2,1e-1,1,1e1]
        #alphas = [0]
        set_seed(r)
        start = time.time()
        import copy 
        for alpha in alphas:
            dm1 = copy.deepcopy(dm)
            pseudo_labeler = copy.deepcopy(dm)
            dmself = train_selfdm(trainloader, X.shape[1],y_all.shape[1], pseudo_labeler = pseudo_labeler, sample_imp = 0, prob_reg = 0, model = dm1, num_epochs = 3000, lambc = 0.5, alpha = alpha)
            dmself_alpha_list.append(dmself)
            rev, hamm, tl, auc = test_multi(dmself, X_valid, y_valid)
            alpha_perf.append(hamm)
        dmval_auc = 1e5
        if dmval_auc < min(alpha_perf):
            dmself = copy.deepcopy(dm)
            print('DM Model is Selected.')
        else:
            dmself = dmself_alpha_list[np.argmin(alpha_perf)]
            print(f'Alpha {alphas[np.argmin(alpha_perf)]} is selected.')
        end = time.time()
        rev, hamm, tl, auc = test_multi(dmself, X_test, y_test)
        print('VAT Rev = %.2f' % rev)
        vat_result.append(rev)
        vat_loss.append(hamm)
        vat_softloss.append(tl)
        vat_time.append(end-start)
        vat_auc.append(auc)

    print('DM:')
    print(np.mean(dm_result))
    print(np.std(dm_result)/np.sqrt(r))
    print(np.mean(dm_loss))
    print(np.std(dm_loss)/np.sqrt(r))
    print(np.mean(dm_time))
    print(np.std(dm_time)/np.sqrt(r))
    print(np.mean(dm_softloss))
    print(np.std(dm_softloss)/np.sqrt(r))
    print(np.mean(dm_auc))
    print(np.std(dm_auc)/np.sqrt(r))
    print('Wang:')
    print(np.mean(wang_result))
    print(np.std(wang_result)/np.sqrt(len(wang_result)))
    print(np.mean(wang_loss))
    print(np.std(wang_loss)/np.sqrt(len(wang_loss)))
    print(np.mean(wang_time))
    print(np.std(wang_time)/np.sqrt(len(wang_time)))
    print(np.mean(wang_softloss))
    print(np.std(wang_softloss)/np.sqrt(len(wang_softloss)))
    print(np.mean(wang_auc))
    print(np.std(wang_auc)/np.sqrt(len(wang_auc)))
    print('DMSELF:')
    print(np.mean(dmself_result))
    print(np.std(dmself_result)/np.sqrt(r))
    print(np.mean(dmself_loss))
    print(np.std(dmself_loss)/np.sqrt(r))
    print(np.mean(dmself_time))
    print(np.std(dmself_time)/np.sqrt(r))
    print(np.mean(dmself_softloss))
    print(np.std(dmself_softloss)/np.sqrt(r))
    print(np.mean(dmself_auc))
    print(np.std(dmself_auc)/np.sqrt(r))
    print('SELF+VAT:')
    print(np.mean(vat_result))
    print(np.std(vat_result)/np.sqrt(r))
    print(np.mean(vat_loss))
    print(np.std(vat_loss)/np.sqrt(r))
    print(np.mean(vat_time))
    print(np.std(vat_time)/np.sqrt(r))
    print(np.mean(vat_softloss))
    print(np.std(vat_softloss)/np.sqrt(r))
    print(np.mean(vat_auc))
    print(np.std(vat_auc)/np.sqrt(r))
    print('DMHSIC :')
    print(np.mean(dmhsic_result))
    print(np.std(dmhsic_result)/np.sqrt(r))
    print(np.mean(dmhsic_loss))
    print(np.std(dmhsic_loss)/np.sqrt(r))
    print(np.mean(dmhsic_time))
    print(np.std(dmhsic_time)/np.sqrt(r))
    print(np.mean(dmhsic_softloss))
    print(np.std(dmhsic_softloss)/np.sqrt(r))
    print(np.mean(dmhsic_auc))
    print(np.std(dmhsic_auc)/np.sqrt(r))
    print('SNIPS:')
    print(np.mean(ips_result))
    print(np.std(ips_result)/np.sqrt(r))
    print(np.mean(ips_loss))
    print(np.std(ips_loss)/np.sqrt(r))
    print(np.mean(ips_time))
    print(np.std(ips_time)/np.sqrt(r))
    print(np.mean(ips_auc))
    print(np.std(ips_auc)/np.sqrt(r))



