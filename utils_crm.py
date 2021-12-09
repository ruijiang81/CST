import scipy.stats as stats
from scipy.special import expit
from random import seed
from random import randrange
import pandas as pd
import scipy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle

price_grid = np.array([1,2,3,4,5,6,7,8,9,10])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# similar synthetic dataset in https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LopezR.787.pdf
def log_prob(X, if_def = 0, price_grid = np.array([1,2,3,4,5,6,7,8,9,10]), biased = 3, ovp = 0):
    if if_def:
        prob = np.array([X[:,i]/np.sum(X[:,2:len(price_grid)],axis=1) for i in range(len(price_grid))]).T
        prob[:,0] = 0
        prob[:,1] = 0
    else:
        if ovp == 0:
            prob = np.array([X[:,i]/np.sum(X[:,:len(price_grid)],axis=1) for i in range(len(price_grid))]).T
        else:
            from scipy.special import softmax
            print(f'ovp is {ovp}')
            prob = softmax(X[:,:len(price_grid)]*ovp,axis=1)  
            print(prob) 
        #prob = np.array([X[:,i]/np.sum(X[:,:len(price_grid)],axis=1) for i in range(len(price_grid))]).T
        #prob = np.array([X[:,i]/np.sum(X[:,:len(price_grid)],axis=1) for i in range(len(price_grid))]).T
    return prob 

def sample_x(n, dim):
    return np.random.uniform(low=0,high=1,size=(n,dim))
    #return np.random.normal(loc = 0.5, scale = 100, size=(n,dim))

def sample_treatment(X, if_def, biased, price_grid = np.array([1,2,3,4,5,6,7,8,9,10]), ovp = 0):
    prob = log_prob(X, if_def, price_grid = price_grid, biased = biased, ovp = ovp)
    treatment = []
    for i in range(X.shape[0]):
        treatment.append(np.random.choice(price_grid, p = prob[i,:]))
    print(treatment)
    return np.array(treatment)

def sample_treatment_withlog(X, if_def, biased, price_grid = np.array([1,2,3,4,5,6,7,8,9,10])):
    prob = log_prob(X, if_def, price_grid = price_grid)
    treatment = []
    logp = []
    for i in range(X.shape[0]):
        action = np.random.choice(range(len(price_grid)), p = prob[i,:])
        treatment.append(price_grid[action])
        logp.append(prob[i,action])
    return np.array(treatment), np.array(logp)

def sample_treatment_withfulllog(X, if_def, biased, price_grid = np.array([1,2,3,4,5,6,7,8,9,10])):
    prob = log_prob(X, if_def, price_grid = price_grid)
    treatment = []
    logp = []
    for i in range(X.shape[0]):
        action = np.random.choice(range(len(price_grid)), p = prob[i,:])
        treatment.append(price_grid[action])
        logp.append(prob[i,action])
    return np.array(treatment), np.array(logp), prob 

def h(x, a, b, c):
    s = 0
    for i in range(c.shape[0]):
        s += a[i] * np.exp(sum(- b[i,:] * np.abs(x-c[i,:])))
    return s/0.008

def stepwise(x):
    if x < 0.1:
        return 0.7
    if 0.1 <= x < 0.3:
        return 0.5
    if 0.3 <= x < 0.6:
        return 0.3
    if 0.6 <= x <= 1:
        return 0.1

def stepwise_inter(x,y):
    if x < 0.1:
        if y > 0.5:
            return 0.65
        else:
            return 0.45
    if 0.1 <= x < 0.3:
        if y > 0.5:
            return 0.55
        else:
            return 0.35
    if 0.3 <= x < 0.6:
        if y > 0.5:
            return 0.45
        else:
            return 0.25
    if 0.6 <= x <= 1:
        if y > 0.5:
            return 0.35
        else:
            return 0.15

def sample_y(x, hx,treatment, dataset):
    if dataset == 1:
        p = expit(hx -  x[:,0] * treatment*2)
        # try new 
    elif dataset == 2:
        p = expit((x[:,0]-0.5)*5 -  0.4 * treatment)
    elif dataset == 3:
        # stepwise
        stepfunc = np.vectorize(stepwise)
        p = expit(hx -  stepfunc(x[:,0]) * treatment * 2)
    elif dataset == 4:
        stepfunc = np.vectorize(stepwise_inter)
        p = expit(hx -  stepfunc(x[:,0],x[:,1]) * treatment * 2)
    elif dataset == 5:
        p = expit(hx - (x[:,0]+x[:,1]) * treatment)
    elif dataset == 6:
        p = expit(hx -  x[:,0] * treatment*0.25)
    y = np.array([np.random.binomial(size=1, n=1, p=i)[0] for i in p])
    return y

def sample_ally(x, hx, dataset, price_grid):
    y = []
    for treatment in price_grid:
        if dataset == 1:
            p = expit(hx -  x[:,0] * treatment * 2)
        elif dataset == 2:
            p = expit((x[:,0]-0.5)*5 -  0.4 * treatment)
        elif dataset == 3:
            # stepwise
            stepfunc = np.vectorize(stepwise)
            p = expit(hx -  stepfunc(x[:,0]) * treatment * 2)
        elif dataset == 4:
            stepfunc = np.vectorize(stepwise_inter)
            p = expit(hx -  stepfunc(x[:,0],x[:,1]) * treatment * 2)
        elif dataset == 5:
            p = expit(hx - (x[:,0]+x[:,1]) * treatment)
        elif dataset == 6:
            p = expit(hx -  x[:,0] * treatment * 0.25)
        y.append([np.random.binomial(size=1, n=1, p=i)[0] for i in p])
        #y.append((p<0.5)*1)
    y = np.array(y).T
    return y
    
def sample_a(dim1, dim2):
    # dim1 for i, dim2 for j 
    a = np.random.uniform(low=0,high=1,size=dim1)
    b = np.random.uniform(low=0,high=1,size=(dim1,dim2))
    c = np.random.uniform(low=0,high=1,size=(dim1,dim2))
    return a, b, c

def prob_a_prime(a, y, price_grid):
    # P(a'|a, y)
    if y == 1:
        prob = [1/(a-min(price_grid)+1) if i+1 <= a else 0 for i in range(len(price_grid)) ]
    if y == 0:
        prob = [1/(max(price_grid)+1-a) if i+1 >= a else 0 for i in range(len(price_grid)) ]
    return np.array(prob)

def prob_a_prime_kdd(a, y):
    # P(a'|a, y)
    prob = [1/2 if i in [0,1] else 0 for i in range(len(price_grid))]
    return np.array(prob)

def prob_y_prime(y):
    return np.array([1 if i==y else 0 for i in range(2)])

def augment_data(X, treatment, y, weights, a, b, c, if_def = 0, rep = 2, dataset = 1):
    # x ~ D, a ~ log(a|x), y ~ binomial(x, a),
    # augmenting, a' ~ P(a'|a, y), y' ~ P(y'|y)
    augment_X = []
    augment_y = []
    augment_treatment = []
    cn4 = 0
    for r in range(rep):
        for i in range(X.shape[0]):
            augment_X.append(X[i,:])
            a_prime = np.random.choice(price_grid, p=prob_a_prime(treatment[i],y[i], price_grid))
            y_prime = y[i]
            augment_y.append(y_prime)
            augment_treatment.append(a_prime)
    augment_X = np.array(augment_X)
    augment_y = np.array(augment_y)
    augment_treatment = np.array(augment_treatment)
    
    #X = np.append(X, augment_X,axis = 0)
    #y = np.append(y, augment_y)
    #treatment = np.append(treatment, augment_treatment)
    X = augment_X
    y = augment_y
    treatment = augment_treatment
    weights = get_aug_weights(X, y, treatment, a, b, c, if_def = if_def, dataset = dataset)  
    return X, y, treatment, weights

def augment_data_kdd(X, treatment, y, dm, weights, a, b, c, impute = 'all', if_def = 0, rep = 5):
    # x ~ D, a ~ log(a|x), y ~ binomial(x, a),
    # augmenting, a' ~ P(a'|a, y), y' ~ P(y'|y)
    augment_X = []
    augment_y = []
    augment_treatment = []
    cn4 = 0
    for r in range(rep):
        for i in range(X.shape[0]):
            augment_X.append(X[i,:])
            a_prime = np.random.choice(price_grid, p=prob_a_prime_kdd(treatment[i],y[i]))  
            if impute == 'all':
                out = dm(torch.cat((torch.from_numpy(X[i,:]), torch.Tensor([a_prime]).double())).float())
                y_prime = torch.nn.Softmax(dim=1)(out.reshape(1,-1))[0][1].detach().numpy() 
                y_prime = np.random.binomial(1,p=y_prime)
            else:
                if y[i] == 1 and a_prime <= treatment[i]:
                    y_prime = 1
                elif y[i] == 0 and a_prime >= treatment[i]:
                    y_prime = 0
                else:
                    out = dm(torch.cat((torch.from_numpy(X[i,:]), torch.Tensor([a_prime]).double())).float())
                    y_prime = torch.nn.Softmax(dim=1)(out.reshape(1,-1))[0][1].detach().numpy() 
                    y_prime = np.random.binomial(1,p=y_prime)
            augment_y.append(y_prime)
            augment_treatment.append(a_prime)
    augment_X = np.array(augment_X)
    augment_y = np.array(augment_y)
    augment_treatment = np.array(augment_treatment)
    weights = get_aug_weights_kdd(augment_X, augment_y, augment_treatment, a, b, c, if_def = if_def) 
    #X = np.append(X, augment_X,axis = 0)
    #y = np.append(y, augment_y)
    #treatment = np.append(treatment, augment_treatment)
    #weights = get_aug_weights(X, y, treatment, a, b, c)  
    X = augment_X
    y = augment_y
    treatment = augment_treatment
    return X, y, treatment, weights

def get_aug_weights(aug_x, aug_y, aug_treatment, a, b, c, if_def = 0, dataset=1):
    # assume propensity score is uniform and deterministic demand function 
    weights = []
    for i in range(aug_x.shape[0]):
        pm = 0
        hx_ = h(aug_x[i,:],a,b,c)
        # marginalize out a and y
        for p in price_grid:
            for y in [0,1]:
                if dataset == 1:
                    dp = expit(hx_ -  aug_x[i:(i+1),0] * p * 2)
                elif dataset == 2:
                    dp = expit((aug_x[i:(i+1),0]-0.5)*5 -  0.4 * p)
                elif dataset == 3:
                    # stepwise
                    stepfunc = np.vectorize(stepwise)
                    dp = expit(hx_ -  stepfunc(aug_x[i:(i+1),0]) * p * 2)
                elif dataset == 4:
                    stepfunc = np.vectorize(stepwise_inter)
                    dp = expit(hx_ -  stepfunc(aug_x[i:(i+1),0],aug_x[i:(i+1),1]) * p * 2)
                elif dataset == 5:
                    dp = expit(hx_ - (aug_x[i:(i+1),0]+aug_x[i:(i+1),1]) * p)
                if aug_y[i] == 1:
                    pm += log_prob(aug_x[i:(i+1),:], if_def)[0][p - 1] * dp * prob_a_prime(p,y, price_grid)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
                else:
                    pm += log_prob(aug_x[i:(i+1),:], if_def)[0][p - 1] * dp * prob_a_prime(p,y, price_grid)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
        pm = pm + 1e-6
        #hx_ = (hx_ - np.mean(hx))/np.std(hx)
        #if aug_y[i] == 1:
        #    pm = pm / (expit(hx_ - 3 * aug_treatment[i] / 5) + 1e-6)
        #else:
        #    pm = pm / (1-expit(hx_ - 3 * aug_treatment[i] / 5) + 1e-6)
        weights.append(pm)
    return np.array(weights)

def get_aug_weights_kdd(aug_x, aug_y, aug_treatment, a, b, c, if_def = 1):
    # assume propensity score is uniform and deterministic demand function 
    weights = []
    for i in range(aug_x.shape[0]):
        pm = 0
        hx_ = h(aug_x[i,:],a,b,c)
        # marginalize out a and y
        #for p in price_grid:
        #    for y in [0,1]:
        #        if aug_y[i] == 1:
        #            pm += log_prob(aug_x[i:(i+1),:])[0][p - 1] * expit(hx_ - 3 * p / 5) * prob_a_prime(p,y)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
        #        else:
        #            pm += log_prob(aug_x[i:(i+1),:])[0][p - 1] * (1-expit(hx_ - 3 * p / 5)) * prob_a_prime(p,y)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
        #pm = pm + 1e-6
        pm = 1 / 2 # len(price_grid)
        #hx_ = (hx_ - np.mean(hx))/np.std(hx)
        #if aug_y[i] == 1:
        #    pm = pm / (expit(hx_ - 3 * aug_treatment[i] / 5) + 1e-6)
        #else:
        #    pm = pm / (1-expit(hx_ - 3 * aug_treatment[i] / 5) + 1e-6)
        weights.append(pm)
    return np.array(weights)

def get_aug_weights_lower(aug_x, aug_y, aug_treatment, a, b, c, if_def = 0):
    # assume propensity score is uniform and deterministic demand function 
    weights = []
    for i in range(aug_x.shape[0]):
        pm = 0
        hx_ = h(aug_x[i,:],a,b,c)
        # marginalize out a and y
        for p in price_grid:
            for y in [0,1]:
                if aug_y[i] == 1:
                    pm += log_prob(aug_x[i:(i+1),:], if_def)[0][p - 1] * prob_a_prime(p,y)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
                else:
                    pm += log_prob(aug_x[i:(i+1),:], if_def)[0][p - 1] * prob_a_prime(p,y)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
        pm = pm + 1e-6
        #hx_ = (hx_ - np.mean(hx))/np.std(hx)
        weights.append(pm)
    return weights

def get_aug_weights_est(aug_x, aug_y, aug_treatment, dm, a, b, c, demand, if_def = 0):
    # assume propensity score is uniform and deterministic demand function 
    weights = []
    for i in range(aug_x.shape[0]):
        pm = 0
        hx_ = h(aug_x[i,:],a,b,c)
        # marginalize out a and y
        for p in price_grid:
            for y in [0,1]:
                out = demand(torch.cat((torch.from_numpy(aug_x[i,:]), torch.Tensor([p]).double())).float())
                prediction = torch.nn.Softmax(dim=1)(out.reshape(1,-1))[0][1].detach().numpy()
                if aug_y[i] == 1:
                    pm += log_prob(aug_x[i:(i+1),:], if_def)[0][p - 1] * prediction * prob_a_prime(p,y)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
                else:
                    pm += log_prob(aug_x[i:(i+1),:], if_def)[0][p - 1] * (1-prediction) * prob_a_prime(p,y)[aug_treatment[i] - 1] * prob_y_prime(y)[aug_y[i]] 
        #hx_ = (hx_ - np.mean(hx))/np.std(hx)
        out = demand(torch.cat((torch.from_numpy(aug_x[i,:]), torch.Tensor([aug_treatment[i]]).double())).float())
        prediction = torch.nn.Softmax(dim=1)(out.reshape(1,-1))[0][1].detach().numpy()
        #if aug_y[i] == 1:
        #    pm = pm / prediction
        #else:
        #    pm = pm / (1-prediction)
        weights.append(pm)
    return weights


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

'''
class RBFkernel():
    def __init__(self, sigma=0.5):
        if torch.cuda.is_available():
            self.sigma = torch.Tensor([sigma]).cuda()
        else:
            self.sigma = torch.Tensor([sigma])
        
    def __call__(self, x, y):
        #numerator = -1 * np.linalg.norm(x - y, ord=2)**2
        if torch.cuda.is_available():
            self.sigma.cuda()
        numerator = -1 * torch.square(torch.norm((x - y).float()))
        denominator = torch.square(self.sigma)
        return torch.exp(numerator / denominator)
'''
             
class RBFkernel():
    def __init__(self, sigma=0.5):
        self.sigma = sigma 

    def __call__(self, x, y):
        #numerator = -1 * np.linalg.norm(x - y, ord=2)**2
        numerator = -1 * torch.square(torch.norm((x - y).float()))
        denominator = self.sigma ** 2
        return torch.exp(numerator / denominator)

            
def gram_matrix(kernel, data, m):
    """
    Arguments:
    =========
    - kernel : kernel function 
    - data : data samples, shape=(m, dim(data_i))
    - m : number of samples
    """
    gram_matrix = torch.zeros((m, m))
    if torch.cuda.is_available():
        gram_matrix = torch.zeros((m, m)).cuda()
    #for i in range(m):
    #    for j in range(m):
    #        gram_matrix[i][j] = kernel(data[i], data[j])
    gram_matrix = torch.mm(data, data.t())
    return gram_matrix


def hsic(k, l, m, X, Y):
    """
    Arguments:
    =========
    - k : kernel function for X
    - l : kernel function for Y
    - m : number of samples
    - X : data samples, shape=(m, dim(X_i))
    - Y : data samples, shape=(m, dim(Y_i))
    """
    H = torch.full((m, m), -(1/m)) + torch.eye(m)
    if torch.cuda.is_available():
        H.to(device='cuda:0')
    K = gram_matrix(k, X, m)
    #print("Gram(X) :", K, "\nGram(X) mean :", K.mean())
    L = gram_matrix(l, Y, m)
    if torch.cuda.is_available():
        L = L.cuda() 
    #print("Gram(Y) :", L, "\nGram(Y) mean :", L.mean())
    a = (K.reshape(-1) * L.reshape(-1)).mean()
    b = (torch.mm(K.reshape(-1,1) , L.reshape(1,-1))).reshape(-1).mean()
    c = torch.bmm(K.view(m,m,1),L.view(m,1,m)).mean()
    HSIC = a+b-2*c
    return HSIC
