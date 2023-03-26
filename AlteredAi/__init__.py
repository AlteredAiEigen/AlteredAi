from AlteredAi.Torch.functions import TorchDataLoader
from AlteredAi.core.AlteredAiDataLoader import AlteredAiDataLoader

def dataloaderApi(access_key_id,secret_access_key,dataKey,dtype=None):

    '''
    :param access_key_id:
    :param secret_access_key:
    :param dataKey:
    :param dtype: tensorflow.tensor , torch.tensor
    :return: dataLoader torch or tensorflow
    '''

    if dtype=='torch.tensor':
        data_object = TorchDataLoader(access_key_id=access_key_id, secret_access_key=secret_access_key, dataKey=dataKey)
        return data_object.getPytorchDataLoader()

    if dtype=='tensorflow.tensor':
        #data_object = TensorflowDataLoader(access_key_id=access_key_id, secret_access_key=secret_access_key, dataKey=dataKey)
        #return data_object.getPytorchDataLoader()
        print("TensorflowDataLoader is not implemented yet")
        return None
    raise Exception("invalid dtype got ",dtype," but valid options are tensorflow.tensor or torch.tensor")

def getDataAsNumpyArrayApi(access_key_id,secret_access_key,dataKey):
    data_=AlteredAiDataLoader(access_key_id,secret_access_key,dataKey)
    return data_.getDataAsNumpyArrays()







