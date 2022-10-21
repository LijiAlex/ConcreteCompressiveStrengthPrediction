from concrete_src.entity.config_entity import DataTransformationConfig
from concrete_src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact, DataIngestionArtifact
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.util.util import read_yaml_file
from concrete_src.constant import *
from concrete_src.util.util import save_numpy_array_data, save_object

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from kneed import KneeLocator

import os, sys
import numpy as np
import pandas as pd

class OutlierImputer(BaseEstimator, TransformerMixin):
    '''
    This class extends the functionality of KNNImputer to handle outliers.
    It makes the outlier null and then uses KNNImputer to fill those values
    '''
    def __init__(self):
        self.a = []
        self.b = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            logging.info("Converting outliers to NaN")
            for i in X.columns:
                q1, q2, q3 = X[i].quantile([0.25,0.5,0.75])
                IQR = q3 - q1
                self.a = X[i] > q3 + 1.5*IQR
                self.b = X[i] < q1 - 1.5*IQR
                X[i] = np.where(self.a | self.b, np.NaN, X[i])  
            return X
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X, y=None)

class ClusterGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.number_of_clusters = 0
        logging.info("Begin clustering")

    def fit(self, X, y=None):
        try:
            self.number_of_clusters = self.get_no_of_clusters(X)
            logging.info(f"Optimal no. of clusters: {self.number_of_clusters}")
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def transform(self, X, y=None):
        try:
            return self.create_clusters(X)
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X, y=None)

    def get_no_of_clusters(self, data):
        try:
            inertias=[] # initializing an empty list to store inertias for each cluster
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                inertias.append(kmeans.inertia_)
            # finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(range(1, 11), inertias, curve='convex', direction='decreasing')
            return self.kn.knee
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def create_clusters(self, data):    
        try:
            kmeans = KMeans(n_clusters=self.number_of_clusters, init='k-means++', random_state=42)
            clusters = kmeans.fit_predict(data) #  divide data into clusters
            data = pd.DataFrame(data)
            data['cluster']=clusters  # create a new column in dataset for storing the cluster information
            logging.info(f"Clusters created.Unique clusters: {data['cluster'].unique()}")
            return data.to_numpy()
        except Exception as e:
            raise ConcreteException(e, sys) from e

class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig, 
    data_validation_artifact: DataValidationArtifact, data_ingestion_artifact: DataIngestionArtifact):
        logging.info(f"{'*'*20}Data Transformation{'*'*20}")
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        self.data_ingestion_artifact = data_ingestion_artifact
        schema_file_path = self.data_validation_artifact.schema_file_path
        self.dataset_schema = read_yaml_file(schema_file_path)

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            # reading schema file path to get target column
            target_column = self.dataset_schema[SCHEMA_FILE_TARGET_COLUMNS]

            numerical_columns = []

            all_columns = self.dataset_schema[SCHEMA_FILE_COLUMNS_KEY]

            for column in all_columns.keys():
                if all_columns[column] !="category" and column != target_column:
                    numerical_columns.append(column)

            logging.info(f"target_column = [{target_column}]")
            logging.info(f"numerical_columns = [{numerical_columns}]")

            data_pipeline = Pipeline([
                ('outlier_imputer', OutlierImputer()),
                ('nan_imputer', KNNImputer(n_neighbors = 3)),
                ('log_transformation', FunctionTransformer(np.log1p)),
                ('std_scaler', StandardScaler()),
                ('cluster_generator', ClusterGenerator())                
            ])

            preprocessing = ColumnTransformer([
                ('column_pipeline', data_pipeline, numerical_columns)
            ])
            return preprocessing        

        except Exception as e:
            raise ConcreteException(e, sys) from e


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            
            logging.info(f"Loading training as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)

            target_column_name = self.dataset_schema[SCHEMA_FILE_TARGET_COLUMNS]

            logging.info(f"Splitting input and target feature from training dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_columns = list(input_feature_train_df.columns)

            logging.info(f"Applying preprocessing object on input features dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            logging.info(f"Stitching back training data frame.")
            input_columns.append('cluster')
            train_df = pd.DataFrame(input_feature_train_arr, columns= input_columns)
            train_df[target_column_name] = target_feature_train_df
            

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            os.makedirs(transformed_train_dir, exist_ok=True)

            train_file_name = os.path.basename(train_file_path)

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)

            logging.info(f"Saving transformed training data at {transformed_train_file_path}")
            
            train_df.to_csv(transformed_train_file_path, index=False, header=True)
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def __del__(self):
        """
        Acts as destructor. Called before all references to the class object are deleted.
        """
        logging.info(f"{'*' *25} Data Transformation log completed {'*' *25}")