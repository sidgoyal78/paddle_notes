
## Inference + training together (using imperative fluid)

Looking at #7464 , let's try to come up with an example in fluid, which potentially might support online training and inference: 

```python
r = fluid.data_getter(raw_input) #getting data from command line 

W = fluid.Tensor()
b = fluid.Tensor()

with fluid.While(r.has_next()):
    mb, is_inference = r.get_next()
    fluid.if is_inference == True:
        x = fluid.layer.data(mb.image)
        y = fluid.layer.fc(x, W, b)
        fluid.print(y)
        
    fluid.else: # we should train
        x = fluid.layer.data(mb.image)
        l = fluid.layer.data(mb.label)
        y = fluid.layer.fc(x, W, b)
        cost = fluid.layer.mse(y, l)
        fluid.optimize(cost)
    
```
