from numpy import load
import boto3
import numpy as np
from .utilities import LoadFile

class AlteredAiDataLoader:
    
    def __init__(self,access_key_id,secret_access_key,dataKey):
        dataKey=dataKey+".npz"
        bucket='alterdai-ai-chest-xray'
        botoObject = boto3.resource(service_name='s3',
                                    region_name='ap-south-1',
                                    aws_access_key_id=access_key_id,
                                    aws_secret_access_key=secret_access_key)
        print("Data Processing job being started")
        print("--------------------------Data Processing Started-------------------------------")
        dataObject = botoObject.Object(bucket_name=bucket, key=dataKey)
        data = LoadFile(dataObject)
        
        # load dict of arrays

        dict_data = load(data)
        # extract the first array
        self.data = dict_data['arr_0']
        print("--------------------------Data Processing Completed-------------------------------")
        # print the array
        print("(N,H,W,C):",self.data.shape)
        print("---------------------------------------------------------------------------------")
        print("N = Size")
        print("H = Height")
        print("W = Width")
        print("C = Channel")
        self.targets = list(np.random.randint(2, size=(len(self.data))))
        
    def getDataAsNumpyArrays(self):
        return self.data,self.targets