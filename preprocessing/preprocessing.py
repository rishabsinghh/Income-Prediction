import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import KNNImputer
import pickle

class preprocessor:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
    def remove_columns(self,data,columns):
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(columns,axis=1)
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()
    def separate_label_feature(self,data,target):
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(target,axis=1)
            self.Y=data[target]
            return self.X,self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()
    def replace_invalid_values(self,data):
        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data
    def encodeCategoricalVariables(self,data):
        data['sex'] = data['sex'].map({' Female': 0,' Male': 1})
        data['salary']=data['salary'].map({" <=50K":0,' >50K':1})
        columns=['sex','salary']
        cat_variables=[feature for feature in data.columns if data[feature].dtype=='O' and feature not in columns]
        encoder=LabelEncoder()
        for feature in cat_variables:
            data[feature+"1"]=encoder.fit_transform(data[feature])
        data=data.drop(cat_variables,axis=1)
        with open('EncoderPickle/enc.pickle', 'wb') as file:
            pickle.dump(encoder, file)
        return data
    def encodeCategoricalvariablespred(self,data):
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})
        columns=['sex']
        cat_variables=[feature for feature in data.columns if data[feature].dtype=='O' and feature not in columns]
        encoder=pickle.load("enc.pickle","wb")
        for feature in cat_variables:
            data[feature+"1"]=encoder.fit_transform(data[feature])
        data=data.drop(cat_variables)
        return data
    def handle_imbalanced_data(self,X,Y):
        rdsmple = RandomOverSampler()
        x_sampled, y_sampled = rdsmple.fit_resample(X, Y)
        return x_sampled,y_sampled
    def is_null_present(self,data):
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data):
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            # rounding the value because KNNimputer returns value between 0 and 1, but we need either 0 or 1
            np.savetxt("sample.txt", self.new_array, delimiter =", ")
            self.new_data=pd.DataFrame(data=np.round(self.new_array), columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()





        
     