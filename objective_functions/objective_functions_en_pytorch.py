import pytorch

def l21_norm(W): 
    return torch.sum(torch.linalg.norm(W, dim=1))

def get_group_regularization(mlp_model):
    const_coeff = lambda W: torch.sqrt((W.get_shape().as_list()[1]).type(torch.FloatTensor))
    return torch.sum([torch.multiply(const_coeff(W), l21_norm(W)) for W in mlp_model.trainable_variables[::] if 'bias' not in W.name])

def sparse_group_lasso(mlp_model):
    grouplasso = get_group_regularization(mlp_model)
    l1 = torch.linalg.norm(mlp_model, dim=1, ord=1)
    sparse_lasso = grouplasso + l1
    return sparse_lasso

def f1_norm(feat, label, mlp_model):
    cross_entropy_norm = torch.mean(torch.categorical_crossentropy(label, mlp_model(feat)))
    return cross_entropy_norm

def f2_norm(mlp_model):
    s_g_l = sparse_group_lasso(mlp_model)
    sparse_group_lasso_norm = s_g_l
    return sparse_group_lasso_norm