from AlteredAi.Torch.functions import TorchDataLoader
from AlteredAi.core.AlteredAiDataLoader import AlteredAiDataLoader
import ivy
def dataloaderApi(access_key_id,secret_access_key,dataKey,dtype=None,resize=64,batchSize=5):

    '''
    :access_key_id:
    :secret_access_key:
    :dataKey:
    :dtype: tensorflow.tensor , torch.tensor
    :resize:64 (default)
    :batchSize:5 (default)
    :return: dataLoader torch or tensorflow
    '''

    if dtype=='torch.tensor':
        data_object = TorchDataLoader(access_key_id=access_key_id, secret_access_key=secret_access_key, dataKey=dataKey,resize=64,batchSize=5)
        return data_object.getPytorchDataLoader()

    if dtype=='tensorflow.tensor':
        #data_object = TensorflowDataLoader(access_key_id=access_key_id, secret_access_key=secret_access_key, dataKey=dataKey)
        #return data_object.getPytorchDataLoader()
        raise Exception("TensorflowDataLoader is not implemented yet")


    raise Exception("invalid dtype got ",dtype," but valid options are tensorflow.tensor or torch.tensor")

def getDataAsNumpyArrayApi(access_key_id,secret_access_key,dataKey):
    data_=AlteredAiDataLoader(access_key_id,secret_access_key,dataKey)
    return data_.getDataAsNumpyArrays()

def getDataAsIvyArrayApi(access_key_id,secret_access_key,dataKey):
    data_ = AlteredAiDataLoader(access_key_id, secret_access_key, dataKey)
    data,targets=data_.getDataAsNumpyArrays()
    return ivy.array(data),ivy.array(targets)







