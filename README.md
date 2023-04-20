![lockup white backgroud color](https://user-images.githubusercontent.com/128656547/227750593-f065d842-e839-4690-8bcb-8f1940ddace1.png)




#                 



https://alteredai.care

- we provide apis to get data directly in form of numpy array and Tensors (pytorch & tensorflow), which can be feeded directly to Algorithms for training.
- No need to store / load data from folders / files for pre-processing.
- Basic essential preprocessing is already done.
- More preprocessing can also be done directly on these provided Tensors with AlteredAi or Torch or Tensorflow or Ivy.
- ML code written with  AlteredAi can be converted to any framework. We are building on top of ivy.

## Install following  

``` 
pip install git+https://github.com/AlteredAiEigen/AlteredAi.git
pip install git+https://github.com/unifyai/ivy.git 
pip install boto3

```
# Poc 
- This is proof of concept where essentially same code can be run with multiple backend.
- You can see the code for poc by navigating to AlteredAi/Ivy/function.

- poc is calling the same code just with different backend each time.
- whichever backend you want your code to run on , please install that framework first.


```python

from AlteredAi.Ivy.function import poc

poc("torch")

poc("tensorflow")

poc("jax")

```

## Ai Model training with pytorch
- Loading data into pytorch dataLoader directly using ```AlteredAi``` and training autoencoder in pytorch
- One can train really any model once they have data in dataloader
- Soon algorithms will be added in ```AlteredAi.Torch.algorithms``` 
- which will help one to use complex algorithms without actually writing code from scratch in pytorch
- some algorithims / architectures are already provided by pytorch , so those won't be implemented here in AlteredAi
- Although calling those algorithims / architectures with easy will be enabled by ```AlteredAi```

```python

import AlteredAi as ai
import torch
import torch.nn as nn
import torch.optim as optim

# Load Data directly into torchDataloader 
torchDataloader=ai.dataloaderApi(access_key_id='AKIA6ARV4U6MKDU4X24E',
                                  secret_access_key='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',
                                  dataKey="TbNormalNumpy",dtype='torch.tensor',batchSize=10,resize=128 ) 
# define the autoencoder model
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        # encoder layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.tconv4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        
    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x
        
    def decode(self, x):
        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))
        x = self.tconv4(x)
        return x
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x



# create the model, optimizer, and loss function
model = CAE()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, _) in enumerate(torchDataloader):
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        # backward pass and optimization
        loss.backward()
        optimizer.step()
        
        
    print(f'Epoch {epoch}, batch {i}, loss: {loss.item()}')


```

