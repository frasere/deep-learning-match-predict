from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


class NeuralNet:
    
    def __init__(self,train_x,train_y,test_x,test_y,input_dim,output_dim):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def build_classifier(self,hn1,hn2,hn3):
        """create model and compile for training
        
        Args
        -----
        input_dim :int
        output_dim:int
        hidden_layer_sizes:np.ndarray
        
        """
        
        model = Sequential()
        # input layer
        model.add(Dense(units = hn1, activation='relu', input_dim = self.input_dim))
        # hidden layers
        model.add(Dense(hn2, activation='relu'))
        model.add(Dense(hn3, activation='relu'))
        # softmax on outer layer to get probs
        model.add(Dense(units = self.output_dim, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
    
    
    def gridCV(self,parameters):
        # perform grid search CV using KC wrapper function
        
        classifier = KerasClassifier(build_fn = self.build_classifier,verbose=0)
        
        gridsearch = GridSearchCV(estimator = classifier,
                         param_grid = parameters,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs = -1,
                         verbose =0)
        
        # grid search CV requires target variable to NOT be one hot encoded
        gridsearch = gridsearch.fit(self.train_x,self.train_y['FTR_le'])
        
        return gridsearch
        
    
    def fit_classifier(self,compiled_model,epochs,batch_size):
        # perform fit to a built model
        
        model = compiled_model
        # fit model
        model.fit(self.train_x,
                  self.train_y[['A','D','H']],
                  epochs=epochs,   
                  batch_size=batch_size,
                  verbose=0,
                 use_multiprocessing=True)
        
        return model
    
    
    def predictions(self,fitted_classifier):
        # return predictions for test set and fitted model
        
        # pred prob and classes
        pred_prob = pd.DataFrame(fitted_classifier.predict_proba(self.test_x),columns=['PA','PD','PH'])
        pred_class = pd.DataFrame(fitted_classifier.predict_classes(self.test_x),columns=['PClass'])
        
        # make a dataframe
        predictions = pred_class.join(pred_prob,lsuffix=True,rsuffix=True)
        final_predictions = self.test_y.reset_index(drop=True).join(predictions)
    
        return final_predictions
    
