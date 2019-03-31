import numpy as np
import matplotlib.pyplot as plt
import utils_linear


data_dir = 'data/'
plot_path = ''

kernal_size = np.array([8,8])
filter_num = 8
n_class = 10
epochs = 22

eta = 0.05
data_shuffle = True
batch_size = 32
rseed = 2019
acfun_name = 'tanh'

conv2d_w_factor = 0.01
mlp1_w_factor = 0.001
mlp2_w_factor = 0.002


acfun = utils_linear.get_acfun(acfun_name)
exp_name = 'kernal_size: '+str(kernal_size)+' filter_num: '+str(filter_num)+' learning rate: '+str(eta) \
+' epochs: '+str(epochs)+' Number of classes: '+str(n_class)+' batch size: '+str(batch_size)+ \
' Shuffle data: '+str(data_shuffle)+' Activation: '+ acfun_name
x_train,y_train,x_test,y_test = utils_linear.load_data_part3(data_dir,True)


history,filter_w_k,mlp1_w_k = utils_linear.model_fit1(x_train,y_train,x_test,y_test,kernal_size,filter_num,
                                                 eta,epochs,n_class,batch_size,data_shuffle,acfun,rseed,
                                                 conv2d_w_factor,mlp1_w_factor)
        
history,conv_mat_k,mlp1_w_k,mlp2_w_k = utils_linear.model_fit2(x_train,y_train,x_test,y_test,kernal_size,
                                                             filter_num,eta,epochs,n_class,batch_size,
                                                             data_shuffle,acfun,rseed,conv2d_w_factor,
                                                             mlp1_w_factor,mlp2_w_factor)


utils_linear.plot_history(history)


