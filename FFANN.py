#Sharoz Haseeb & Ahmad Waqas

import numpy as np
import sys

# this is a working implimentation of a feed forward neural net which can be configured to work on any dataset.

# the weight vectors are dynamically initialized by ran_weights variable.


class FFANN(object):
# the hidden layers as specified in the document has been set to 16 by default in h_layers variable.
# the weight vectors are dynamically initialized by ran_weights variable.
# the hidden layers as specified in the document has been set to 16 by default in h_layers variable.
    def __init__(self, h_nodes=30,float1=0., h_layers=16, learningRate=0.001,shuffle=True, samples_size=1, ran_weights=None):self.random = np.random.RandomState(ran_weights)
        self.h_nodes = h_nodes
        self.float1 = float1
        self.h_layers = h_layers
        self.learningRate = learningRate
        self.shuffle = shuffle
        self.samples_size = samples_size
    # the following function will encode labels into one hot encoding     
    def _encoding(self, y, n_classes):

        encoding = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):encoding[val, idx] = 1.
        return encoding.T

    # logistic sigmoid activation function

    def act_func(self, z):

        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))


    def _weight_update(self, X):

        #First we update weights through hidden layers
        z_h = np.dot(X, self.w_h) + self.b_h
        # apply the activation (sigmoid) function on the net input from the hidden layer
        a_h = self.act_func(z_h)


        z_out = np.dot(a_h, self.w_out) + self.b_out
        # apply the activation function again on output.
        a_out = self.act_func(z_out)
        return z_h, a_h, z_out, a_out


        def _mean_sq_error(self, y_enc, output):

        FLOAT_term = (self.float1 *(np.sum(self.w_h ** 2.) +np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + FLOAT_term
        return cost
    

    def predict(self, X):

        z_h, a_h, z_out, a_out = self._weight_update(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def calc_weights(self, X_train, y_train, X_valid, y_valid):

        n_output = np.unique(y_train).shape[0] # number of classes 
        #class labels
        n_features = X_train.shape[1]

        # Random Weight initialization
        self.b_h = np.zeros(self.h_nodes)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
        size=(n_features,
        self.h_nodes))
        # weights of hidden layer to output layer
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
        size=(self.h_nodes,
        n_output))
        epoch_strlen = len(str(self.h_layers)) # for progr. format.
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': \
        []}
        y_train_enc = self._encoding(y_train, n_output)

        # iterate over training h_layers
        for i in range(self.h_layers):

        # iterate over samples
            indices = np.arange(X_train.shape[0])
        if self.shuffle:
        self.random.shuffle(indices)
        for start_idx in range(0, indices.shape[0] -\self.samples_size +\
                       1, self.samples_size):
            batch_idx = indices[start_idx:start_idx +\self.samples_size]

        # forward propagation
            z_h, a_h, z_out, a_out = \self._weight_update(X_train[batch_idx])
            z_h, a_h, z_out, a_out = self._weight_update(X_train)
    
        # calculating the cost to compare different data sets
            cost = self._mean_sq_error(y_enc=y_train_enc,
            output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train ==
            y_train_pred)).astype(np.float1) /
            X_train.shape[0])
            valid_acc = ((np.sum(y_valid ==
            y_valid_pred)).astype(np.float1) /
            X_valid.shape[0])
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
            '| Train/Valid Acc.: %.2f%%/%.2f%% '
            %
            (epoch_strlen, i+1, self.h_layers,
            cost,
            train_acc*100, valid_acc*100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

            return self
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
            output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train ==
            y_train_pred)).astype(np.float1) /
            X_train.shape[0])
            valid_acc = ((np.sum(y_valid ==
            y_valid_pred)).astype(np.float1) /
            X_valid.shape[0])
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
            '| Train/Valid Acc.: %.2f%%/%.2f%% '
            %
            (epoch_strlen, i+1, self.epochs,
            cost,
            train_acc*100, valid_acc*100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            return self

# after training our neural network and now we are goinfg to evaluate it


        z_h, a_h, z_out, a_out = self._forward(X_train)

        cost = self._compute_cost(y_enc=y_train_enc,output=a_out)
        y_train_pred = self.predict(X_train)
        y_valid_pred = self.predict(X_valid)
        
        train_acc = ((np.sum(y_train ==y_train_pred)).astype(np.float1) /X_train.shape[0])

        valid_acc = ((np.sum(y_valid ==y_valid_pred)).astype(np.float1) /X_valid.shape[0])

        sys.stderr.write('\r%0*d/%d | Cost: %.2f '
        '| Train/Valid Acc.: %.2f%%/%.2f%% '
        %
        (epoch_strlen, i+1, self.epochs,cost,train_acc*100, valid_acc*100))

        sys.stderr.flush()
        self.eval_['cost'].append(cost)
        self.eval_['train_acc'].append(train_acc)
        self.eval_['valid_acc'].append(valid_acc)
        return self

# Finally now we initialize our FFANN
nn = FFANN(h_nodes=100,float1=0,h_layers=16,learningRate=0.0005,samples=100,shuffle=True,ran_weights=1)
# To run any dataset load the training and testing data in numpy arrays and pass then to the respective functions.
