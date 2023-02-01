from torch import nn

def save_model_parameters(model):
    parameters_names = list(model.state_dict().keys())  
    params = []
    for m in model.modules():
        for n, _ in m.named_parameters(recurse=False):
            params.append((m, n, 'parameter'))

        for n, _ in m.named_buffers(recurse=False):
            params.append((m, n, 'buffer'))
    
    return params, parameters_names

def load_model_parameters(params, names, weights):
    index = 0
    
    for m, n, t in params:
        if(t == 'parameter'): # Parameter
            delattr(m, n)
            m.register_parameter(n, nn.parameter.Parameter(weights[names[index]]))
            #setattr(m, n, nn.parameter.Parameter(weights[names[index]])) # Also work       
        else: # Buffer
            delattr(m, n) # Delete buffer before re-adding it
            m.register_buffer(n, weights[names[index]], persistent=True) # Add buffer + state_dict
            #setattr(m, n, listed_weights[index]) # Only add buffer
            
        index += 1       