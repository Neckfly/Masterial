#!/usr/bin/python3
#########################################################################
# Author: Tina Issa
# Last updated: October 15, 2021
# This class defiines the objective functions of the neural network. 
#########################################################################

import tensorflow as tf

def l21_norm(W): 
    return tf.reduce_sum(tf.norm(W, axis=1))	#reduce_sum -> sum en pytorch

def get_group_regularization(mlp_model):
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1] ,tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l21_norm(W)) for W in mlp_model.trainable_variables[2::] if 'bias' not in W.name])

def get_L1_norm(mlp_model):		#doit exister en pytorch
    variables = [tf.reshape(v ,[-1]) for v in mlp_model.trainable_variables[2::]]
    variables = tf.concat(variables, axis= 0)
    return tf.norm(variables, ord = 1)

def sparse_group_lasso(mlp_model):
    grouplasso = get_group_regularization(mlp_model)
    l1 = get_L1_norm(mlp_model)
    sparse_lasso = grouplasso + l1
    return sparse_lasso

def f1_norm(feat, label, max_a, mlp_model):	#en pytorch : crossentropie
    cross_entropy_norm = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label, mlp_model(feat))) / max_a	#reduce_mean -> mean en pytorch
    return cross_entropy_norm

def f2_norm(max_b, mlp_model):
    s_g_l = sparse_group_lasso(mlp_model)
    sparse_group_lasso_norm = s_g_l/ max_b
    return sparse_group_lasso_norm

    