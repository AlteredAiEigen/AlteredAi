import ivy

class MyModel(ivy.Module):
    def __init__(self):
        self.linear0 = ivy.Linear(3, 64)
        self.linear1 = ivy.Linear(64, 1)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = ivy.relu(self.linear0(x))
        return ivy.sigmoid(self.linear1(x))

class poc():
       def __init__(self,backend_choice):
           assert isinstance(backend_choice, str)
           self.backend_choice=backend_choice
           ivy.set_backend(backend_choice)  # change to any backend!

           print('training with :' ,backend_choice)
           model = MyModel()
           optimizer = ivy.Adam(1e-4)
           x_in = ivy.array([1., 2., 3.])
           target = ivy.array([0.])


           def loss_fn(v):
               out = model(x_in, v=v)
               return ivy.mean((out - target) ** 2)

           for step in range(100):
               loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
               model.v = optimizer.step(model.v, grads)
               print('Step: {} --- Loss: {}'.format(step, ivy.to_numpy(loss).item()))

           print('Finished training with ',backend_choice,' Now you can change the backend_choice and  try again')
           print('supported backend types are : torch , tensorflow and jax')
