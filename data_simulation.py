# This function defines the simulated datasets used in the simulations
#
# input: - dataset_name- index of dataset to simulate
#
# output: - treatment effect- function defining how change in treatment(price P) will change latent variable
#         - g_x - function defining effect of X on latent variable which does not interact with treatment (see above)
import torch
def define_data(dataset_name):
    if dataset_name== 1:
        def treatment_effect(X):
            tow_x = - 1
            return(tow_x)
        def g_x(X):
            gx = X[:,0] 
            return(gx)
    elif dataset_name== 2:
        def treatment_effect(X):
            tow_x = - X[:,0]
            return(tow_x)
        
        def g_x(X):
            gx = X[:,0] 
            return(gx)
    elif dataset_name==3:
        def treatment_effect(X):
            tow_x=np.ones(len(X))
            for i in range(len(X)):
                if X[i,0]< -1:
                    tow_x[i] = -1.2
                elif X[i,0]< 0:
                    tow_x[i] = -1.1
                elif X[i,0]< 1:
                    tow_x[i] = -0.9
                else :
                    tow_x[i] = -0.8
            return(tow_x)
        def g_x(X):
            gx = 5
            return(gx)
    elif dataset_name==4: 
        def treatment_effect(X):
            tow_x=np.ones(len(X))
            for i in range(len(X)):
                if X[i,0]< -1:
                    tow_x[i]= -1.25
                    if X[i,1]>0:
                        tow_x[i] = tow_x[i]+0.1
                    if X[i,1]<0:
                        tow_x[i] = tow_x[i]-0.1
                elif X[i,0]< 0:
                    tow_x[i] = -1.1
                    if X[i,1]>0:
                        tow_x[i] = tow_x[i]+0.1
                    if X[i,1]<0:
                        tow_x[i] = tow_x[i]-0.1
                elif X[i,0]< 1:
                    tow_x[i] = -0.9
                    if X[i,1]>0:
                        tow_x[i] = tow_x[i]+0.1
                    if X[i,1]<0:
                        tow_x[i] = tow_x[i]-0.1
                else :
                    tow_x[i] = -0.75
                    if X[i,1]>0:
                        tow_x[i] = tow_x[i]+0.1
                    if X[i,1]<0:
                        tow_x[i] = tow_x[i]-0.1
            return(tow_x)
        def g_x(X):
            gx = 5
            return(gx)
    elif dataset_name==5:
        def treatment_effect(X):
            tow_x = - 1
            return(tow_x)
        def g_x(X):
            gx = X[:,0] 
            return(gx)
    elif dataset_name==6:
        def treatment_effect(X):
            tow_x = - abs(X[:,0]+X[:,1])
            return(tow_x)
        def g_x(X):
            gx = 4*abs(X[:,0]+X[:,1])
            return(gx)
    return treatment_effect,g_x

# This function evaluates the latent variable (response) for a given data set (X) and treatment
def true_latent_function(treatment, X):
    estimate = (treatment_effect(X)*treatment + g_x(X))
    return(estimate)


# This function generates data used in synthetic simulation experiments according to latent varible probit model above
#
# input: - n_data - (n) number of datapoints to generate
#        - n_dimension - (d) dimension of X covariates
#        - dataset_name- index of dataset to simulate
#
# output: - X - (n x d) array of other variables with effect probability of selling
#         - treatment - (n) array of prices
#         - y - (n) array of binary outcome (does item sell)

def train_model_log(model, x, y, num_epochs = 1000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
    if torch.cuda.is_available():
        inputx = Variable(x).cuda()
        targety = Variable(y).cuda()
    else:
        inputx = Variable(x)
        targety = Variable(y)
    for epoch in range(num_epochs):
        # forward
        out = model(inputx.float())
        loss = criterion(out, targety.long())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('log model created.')
    return model 

def get_optimal_price_dm(x, model):
    x = torch.from_numpy(x)
    #price_grid = np.arange(1,9)
    price = price_grid
    opt = []
    for i in x:
        res = []
        for p in price:
            out = model(torch.cat((i, torch.Tensor([p]).double())).float())
            #prediction = torch.argmax(out).numpy()*1
            prediction = torch.nn.Softmax(dim=1)(out.reshape(1,-1))[0][1].detach().numpy()
            res.append(prediction * p)
        #opt.append(np.random.choice(price,1,p=scipy.special.softmax(res))[0])
        opt.append(price[np.argmax(res)])
    return(opt)

def get_optimal_price(x, model):
    x = torch.from_numpy(x)
    #price_grid = np.arange(1,9)
    price = price_grid
    opt = []
    for i in x:
        res = []
        out = model(i.float())
        #out = model(torch.cat((i, torch.Tensor([p]).double())).float())
        prediction = torch.argmax(out).numpy()*1
        #opt.append(np.random.choice(price,1,p=scipy.special.softmax(res))[0])
        opt.append(price[prediction])
    return(opt)

def get_test_data_dm(n_data, n_dimension, dataset_name, model):
    np.random.seed(0)
    if (dataset_name==1):
        X = np.random.normal(loc=5,size = (n_data, n_dimension))
    elif(dataset_name==5):
        X = np.random.normal(loc=5,size = (n_data, n_dimension))
    else:
        X = np.random.normal(size = (n_data, n_dimension))
        
    treatment = get_optimal_price_dm(X, model)
    # softmax sampling 
    treatment = np.array(treatment)
    print(treatment)

    outcomeNoise = np.random.normal(size = n_data, loc= 0,scale= 1)
    
    no_noise_latent_y = true_latent_function(treatment,X)
    latent_y = no_noise_latent_y #+ outcomeNoise
    y = np.zeros(shape = n_data)
    y[latent_y > 0] = 1
    treatment_change=0

    revenue = sum(y * treatment)
    
    return X, treatment, y, revenue

def get_test_data(n_data, n_dimension, dataset_name, model):
    np.random.seed(0)
    if (dataset_name==1):
        X = np.random.normal(loc=5,size = (n_data, n_dimension))
    elif(dataset_name==5):
        X = np.random.normal(loc=5,size = (n_data, n_dimension))
    else:
        X = np.random.normal(size = (n_data, n_dimension))
        
    treatment = get_optimal_price(X, model)
    # softmax sampling 
    treatment = np.array(treatment)
    print(treatment)

    outcomeNoise = np.random.normal(size = n_data, loc= 0,scale= 1)
    
    no_noise_latent_y = true_latent_function(treatment,X)
    latent_y = no_noise_latent_y #+ outcomeNoise
    y = np.zeros(shape = n_data)
    y[latent_y > 0] = 1
    treatment_change=0

    revenue = sum(y * treatment)
    
    return X, treatment, y, revenue

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def generate_data(n_data, n_dimension, dataset_name):
    #price_grid = np.arange(1,9)
    if (dataset_name==1):
        X = np.random.normal(loc=5,size = (n_data, n_dimension))
        #treatment = np.random.uniform(size = n_data, low = 1, high = 8)
        treatment = np.random.choice(price_grid, size = n_data, replace = True, p=[0.9,0.1])
        #treatment = np.random.normal(size = n_data,loc=5, scale=1)
    elif(dataset_name==5):
        X = np.random.normal(loc=5,size = (n_data, n_dimension))
        #treatment = np.random.uniform(size = n_data, low = 1, high = 8)
        treatment = np.random.choice(price_grid, size = n_data, replace = True, p=[0.9,0.1])
        #treatment = np.random.normal(size = n_data,loc=X[:,0], scale=1) 
    else:
        X = np.random.normal(size = (n_data, n_dimension))
        treatment = np.random.choice(price_grid, size = n_data, replace = True, p=[0.9,0.1])
        #treatment = np.random.uniform(size = n_data, low = 1, high = 8)
        #treatment = np.random.normal(size = n_data,loc=5 + X[:,0] , scale=1)
        
    outcomeNoise = np.random.normal(size = n_data, loc= 0,scale= 1)
    
    no_noise_latent_y = true_latent_function(treatment,X)
    latent_y = no_noise_latent_y #+ outcomeNoise
    y = [np.random.binomial(size=1, n=1, p=i)[0] for i in expit(no_noise_latent_y)]
    y = np.array(y)
    treatment_change=0
    return X,treatment,y    

def augment_data_binary(X, treatment, y):
    augment_X = []
    augment_y = []
    augment_treatment = []
    cn1 = 0
    cn2 = 0
    cn3 = 0
    cn4 = 0
    for i in range(X.shape[0]):
        # N1
        if y[i] == 1 and min(price_grid) == treatment[i]:
            augment_X.append(X[i,:])
            augment_y.append(y[i])
            augment_treatment.append(treatment[i])
            cn1 += 1
        # N2
        if y[i] == 0 and max(price_grid) == treatment[i]:
            augment_X.append(X[i,:])
            augment_y.append(y[i])
            augment_treatment.append(treatment[i])
            cn2 += 1
        # if an item didn't sell, higher price won't sell
        # N3
        if y[i] == 0 and min(price_grid) == treatment[i]:
            augment_X.append(X[i,:])
            augment_y.append(y[i])
            augment_treatment.append(max(price_grid))
            cn3 += 1
        # if an item sold, lower price also sells
        # N4
        if y[i] == 1 and max(price_grid) == treatment[i]:
            augment_X.append(X[i,:])
            augment_y.append(y[i])
            augment_treatment.append(min(price_grid))
            cn4 += 1
    print('n4 portion:')
    print(cn4/len(y))
    augment_X = np.array(augment_X)
    augment_y = np.array(augment_y)
    augment_treatment = np.array(augment_treatment)

    X = np.append(X,augment_X,axis = 0)
    y = np.append(y, augment_y)
    treatment = np.append(treatment, augment_treatment)
    return X, y, treatment

def get_aug_weights(aug_x, aug_y, aug_treatment):
    # assume propensity score is uniform and deterministic demand function 
    weights = []
    for i in range(aug_x.shape[0]):
        # I1
        if aug_treatment[i] == min(price_grid) and aug_y[i] == 1:
            pm = 0.9*expit(true_latent_function(np.array([min(price_grid)]),np.array([aug_x[i]]))[0]) + 0.1*expit(true_latent_function(np.array([max(price_grid)]),np.array([aug_x[i]]))[0])*0.5
        # I2
        if aug_treatment[i] == max(price_grid) and aug_y[i] == 0:
            pm = 0.9*(1-expit(true_latent_function(np.array([min(price_grid)]),np.array([aug_x[i]]))[0]))*0.5 + 0.1*(1-expit(true_latent_function(np.array([max(price_grid)]),np.array([aug_x[i]]))[0]))
        # I3
        if aug_treatment[i] == min(price_grid) and aug_y[i] == 0:
            pm = 0.9*(1-expit(true_latent_function(np.array([min(price_grid)]),np.array([aug_x[i]]))[0]))*0.5
        # I4
        if aug_treatment[i] == max(price_grid) and aug_y[i] == 1:
            pm = 0.1*expit(true_latent_function(np.array([max(price_grid)]),np.array([aug_x[i]]))[0])*0.5
        if aug_y[i] == 1:
            pm = pm / expit(true_latent_function(np.array([aug_treatment[i]]),np.array([aug_x[i]]))[0])
        else:
            pm = pm / (1 - expit(true_latent_function(np.array([aug_treatment[i]]),np.array([aug_x[i]]))[0]))
        weights.append(pm)
    return weights

def est_aug_weights(aug_x, aug_y, aug_treatment, model):
    # assume propensity score is uniform and deterministic demand function 
    weights = []
    for i in range(aug_x.shape[0]):
        out1 = model(torch.cat((aug_x[i], torch.Tensor([min(price_grid)]).double())).float())
        pr1 = torch.nn.Softmax(dim=1)(out1.reshape(1,-1))[0].detach().numpy()
        out2 = model(torch.cat((aug_x[i], torch.Tensor([max(price_grid)]).double())).float())
        pr2 = torch.nn.Softmax(dim=1)(out2.reshape(1,-1))[0].detach().numpy()
        # I1
        if aug_treatment[i] == min(price_grid) and aug_y[i] == 1:
            pm = 0.9*pr1[1] + 0.1*pr2[1]*0.5
            pm = pm / pr1[1]
        # I2
        if aug_treatment[i] == max(price_grid) and aug_y[i] == 0:
            pm = 0.9*pr1[0]*0.5 + pr2[0]*0.1
            pm = pm / pr2[0]
        # I3
        if aug_treatment[i] == min(price_grid) and aug_y[i] == 0:
            pm = 0.9*pr1[0]*0.5
            pm = pm / pr1[0]
        # I4
        if aug_treatment[i] == max(price_grid) and aug_y[i] == 1:
            pm = 0.1*pr2[1]*0.5
            pm = pm / pr2[1]
        weights.append(pm)
    return weights
