#lecture 8                                    
import numpy as np

class Tensor:
    
    def __init__(self, data, autograd=False, creators=None, creation_op=None):
        
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        
        self.id = id(self)
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        
        if(creators):
            
            for c in creators:
                
                if(self.id not in c.children):
                    
                    c.children[self.id] = 1
                    
                else:
                    
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        
        for cid, cnt in self.children.items():
            
            if(cnt != 0):
                
                return False
            
        return True        
        
    def backward(self, grad=None, grad_origin=None):
        
        if(self.autograd):
            
            if(not grad):
                
                grad = Tensor(np.ones_like(self.data))
            
            if(grad_origin):
                
                if(self.children[grad_origin.id] == 0):
                    
                    raise Exception("cannot backprop more than once")
                    
                else:
                    
                    self.children[grad_origin.id] -= 1

            if(not self.grad):
                
                self.grad = grad
                
            else:
                
                self.grad += grad
            
            assert grad.autograd == False
            
            if(self.creators and (self.all_children_grads_accounted_for() or not grad_origin)):

                if(self.creation_op == "add"):
                    
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                if(self.creation_op == "neg"):
                    
                    self.creators[0].backward(self.grad.__neg__())
                    
                if(self.creation_op == "sub"):
                    
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if(self.creation_op == "mul"):
                    
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(self.grad * self.creators[0], self)                    
                    
                if(self.creation_op == "transpose"):
                    
                    self.creators[0].backward(self.grad.transpose())

                if("sum" in self.creation_op):
                    
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim, self.creators[0].data.shape[dim]))

                if("expand" in self.creation_op):
                    
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                    
                if(self.creation_op == "mm"):
                    
                    self.creators[0].backward(self.grad.mm(self.creators[1].transpose()))
                    self.creators[1].backward(self.grad.transpose().mm(self.creators[0]).transpose())
                    
    def __add__(self, other):
        
        if(self.autograd and other.autograd):
            
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        
        return Tensor(self.data + other.data)

    def __neg__(self):
        
        if(self.autograd):
            
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        
        if(self.autograd and other.autograd):
            
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        
        if(self.autograd and other.autograd):
            
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="mul")
        
        return Tensor(self.data * other.data)    

    def sum(self, dim):
        
        if(self.autograd):
            
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim,copies):

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if(self.autograd):
            
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        
        return Tensor(new_data)
    
    def transpose(self):
        
        if(self.autograd):
            
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        
        return Tensor(self.data.transpose())
    
    def mm(self, x):
        
        if(self.autograd):
            
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self,x],
                          creation_op="mm")
        
        return Tensor(self.data.dot(x.data))
    
    def __repr__(self):
        
        return str(self.data)                  

