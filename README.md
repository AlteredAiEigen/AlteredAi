# AlteredAi

- we provide apis to get data directly in form of numpy array and Tensors (pytorch & tensorflow), which can be feeded directly to Algorithms for training.
- No need to store data locally for pre-processing.
- Preprocessing can be done directly on these provided Tensors with AlteredAi or Torch or Tensorflow.
- ML code written with  AlteredAi can be converted to any framework. We are building on top of ivy.

## Install
``` pip install git+https://github.com/AlteredAiEigen/AlteredAi.git ```

# Poc 
- This is proof of concept where essentially same code can be run with multiple backend.

- Install following  
``` 
pip install git+https://github.com/AlteredAiEigen/AlteredAi.git
pip install git+https://github.com/unifyai/ivy.git 

```
- poc is calling the same code just with different backends each time.
- whichever backend you want your code to run , please install that framework first.

```

from AlteredAi.Ivy.function import poc

poc("torch")

poc("tensorflow")

poc("jax")



```



