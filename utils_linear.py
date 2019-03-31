import numpy as np
import matplotlib.pyplot as plt

def load_data_part2(data_dir):
    class1_train = reshape_input(np.loadtxt(data_dir+'Part2_1_Train.csv',delimiter=','))
    class2_train = reshape_input(np.loadtxt(data_dir+'Part2_3_Train.csv',delimiter=','))
    class1_test = reshape_input(np.loadtxt(data_dir+'Part2_1_Test.csv',delimiter=','))
    class2_test = reshape_input(np.loadtxt(data_dir+'Part2_3_Test.csv',delimiter=','))
    
    y_class1_train = np.zeros((class1_train.shape[0],2))
    y_class1_train[:,0]+=1
    y_class2_train = np.zeros((class2_train.shape[0],2))
    y_class2_train[:,1]+=1
    y_class1_test = np.zeros((class1_test.shape[0],2))
    y_class1_test[:,0]+=1
    y_class2_test = np.zeros((class2_test.shape[0],2))
    y_class2_test[:,1]+=1
    
    x_train = np.concatenate((class1_train,class2_train))
    y_train = np.concatenate((y_class1_train,y_class2_train))
    
    x_test = np.concatenate((class1_test,class2_test))
    y_test = np.concatenate((y_class1_test,y_class2_test))
    return x_train,y_train,x_test,y_test

def load_data_part3(data_dir,full_data=False):
    if full_data:
        a = 5000
        b = 500
    else:
        a = 1000
        b = 100
    x_train = np.zeros((a,28,28))
    x_test = np.zeros((a,28,28))
    y_train =np.zeros((a,10))
    y_test =np.zeros((a,10))
    
    for n in range(10):
        x_train[(0+n*b):(n+1)*b,:,:] = reshape_input(np.loadtxt(data_dir+'Part3_'+str(n)+'_Train.csv',
               delimiter=','))[range(b),:,:]
        x_test[(0+n*b):(n+1)*b,:,:] = reshape_input(np.loadtxt(data_dir+'Part3_'+str(n)+'_Test.csv',
           delimiter=','))[range(b),:,:]
        y_train[(0+n*b):(n+1)*b,n] = 1
        y_test[(0+n*b):(n+1)*b,n] = 1
    return x_train,y_train,x_test,y_test

def get_conv_matrix(kernal_mat,input_mat):
    rf_size = np.array((1+input_mat.shape[1]-kernal_mat.shape[0],1+input_mat.shape[2]-kernal_mat.shape[1]))
    conv_mat = np.zeros((rf_size[0]*rf_size[1]*kernal_mat.shape[2],input_mat.shape[1]*input_mat.shape[2]))
    conv_mat_bool = conv_mat.copy()
    start_idx_list = np.zeros(rf_size[0]*rf_size[1]*kernal_mat.shape[2])
    for filter_idx in range(kernal_mat.shape[2]):
        row_base = np.array([])
        row_base_bool = row_base.copy()
        for i in range(kernal_mat.shape[0]):
            row_base = np.concatenate((row_base,kernal_mat[i,:,filter_idx],np.repeat(0,rf_size[1]-1)))
            row_base_bool = np.concatenate((row_base_bool,np.ones_like(kernal_mat[i,:,filter_idx]),
                                            np.repeat(0,rf_size[1]-1)))
        row_base=row_base[0:(row_base.shape[0]-(rf_size[1]-1))]
        row_base_bool=row_base_bool[0:(row_base_bool.shape[0]-(rf_size[1]-1))]
        for i in range(rf_size[0]):
            for j in range(rf_size[1]):
                node_idx = filter_idx*rf_size[0]*rf_size[1]+j+i*rf_size[1]
                row_start_idx = i*input_mat.shape[1]+j
                conv_mat[node_idx,row_start_idx:(row_start_idx+row_base.shape[0])] = row_base
                conv_mat_bool[node_idx,row_start_idx:(row_start_idx+row_base.shape[0])] = row_base_bool
                start_idx_list[node_idx] = row_start_idx
                
    l = row_base.shape[0]        
    return conv_mat,start_idx_list,l,conv_mat_bool

def get_filter_matrix(conv_mat,kernal_size,input_shape,filter_num):
    row_idx = int(conv_mat.shape[0]/filter_num)
    filter_w = np.zeros((kernal_size[0],kernal_size[1],filter_num))
    for i in range(filter_num):
        for j in range(kernal_size[0]):
            filter_w[j,:,i] = conv_mat[row_idx*i,(input_shape[1]*j):(input_shape[1]*j+kernal_size[1])]
    return filter_w
            
def conv_transpose(conv_mat,input_mat):
    conv_v = np.matmul(conv_mat,np.reshape(input_mat,(input_mat.shape[0]*input_mat.shape[1],1)))
    deconv_mat = conv_mat.transpose()
    upsample = np.matmul(deconv_mat,np.reshape(conv_v,(conv_v.shape[0]*conv_v.shape[1],1)))
    upsample = np.reshape(upsample,input_mat.shape)
    return upsample

def tanh(x,d=False):
    fx = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if d:
        return 1-(fx**2)
    return fx

def sigmoid(x,d=False):
    fx = 1/(1+np.exp(-x))
    if d:
        return fx*(1-fx)
    return fx

def relu(x,d=False):
    if d:
        return (x>0)+0
    else:
        fx = x
        fx[fx<0] = 0
        return fx

def get_acfun(fun_name):
    if fun_name == 'tanh':
        return tanh
    elif fun_name == 'sigmoid':
        return sigmoid
    elif fun_name == 'relu':
        return relu
    else:
        return 0

def reshape_input(x):
    x_out = np.zeros((x.shape[0],28,28))
    for i in range(x.shape[0]):
        x_out[i,:,:] = np.reshape(x[i,:],(28,28),order='F')
    return x_out

def plot_history(history,filename=''):
    plt.plot(history[:,2])
    plt.plot(history[:,3])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if filename != '':
        plt.savefig(filename+'acc.png')
    else:
        plt.show()
    # summarize history for loss
    plt.plot(history[:,0])
    plt.plot(history[:,1])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    if filename != '':
        plt.savefig(filename+'loss.png')
    else:
        plt.show()

def get_loss(y_true,y_pred):
    return np.sum(np.square(y_true-y_pred))

def get_acc(y_true,y_pred):
    return np.mean(np.argmax(y_true,axis=1) == np.argmax(y_pred,axis=1))

def predict1(x,conv_mat,mlp1_w,acfun):
    x_flatten = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])).transpose()
    conv2d_v = np.matmul(conv_mat,x_flatten)
    conv2d_y = relu(conv2d_v)
    mlp_v = np.matmul(mlp1_w,conv2d_y)
    output = acfun(mlp_v)
    output = output.transpose()
    return output

def back_prop1(x,y_true,conv_mat,mlp1_w,eta,acfun,start_idx_list,l,conv_mat_bool,filter_num):
    # forward pass
    x_flatten = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
    conv2d_v = np.matmul(conv_mat,x_flatten.transpose())
    conv2d_y = relu(conv2d_v)
    mlp_v = np.matmul(mlp1_w,conv2d_y)
    output = acfun(mlp_v).transpose()
    # update output weights:
    err_H_k = y_true-output
    d_H_k = (-1)* np.multiply(err_H_k ,acfun(mlp_v.transpose(),True))
    d_H_k1 = np.reshape(d_H_k,(d_H_k.shape[0],d_H_k.shape[1],1))
    grad_mlp_w_k = np.multiply(d_H_k1,np.reshape(conv2d_y.transpose(),
                                      (conv2d_y.shape[1],1,conv2d_y.shape[0])))
    grad_mlp_w_k1 = (-eta)*np.mean(grad_mlp_w_k,axis = 0)
    mlp1_w_k = mlp1_w + grad_mlp_w_k1
    # update convolution weights:
    d_conv_k = np.multiply(np.matmul(d_H_k,mlp1_w),relu(conv2d_v.transpose(),True))
    grad_conv_k = np.matmul(np.reshape(d_conv_k,(d_conv_k.shape[0],d_conv_k.shape[1],1)),
                            np.reshape(x_flatten,(x_flatten.shape[0],1,x_flatten.shape[1])))
    grad_conv_k1 = (-eta)*np.mean(grad_conv_k,axis = 0)
    grad_conv_k2 = grad_conv_k1 * conv_mat_bool
    
    conv_mat_k = conv_mat.copy()
    row_per_filter = int(grad_conv_k1.shape[0]/filter_num)
    for filter_idx in range(filter_num):
        shared_grad = np.zeros(l)
        row_start_idx = filter_idx*row_per_filter
        row_end_idx = row_start_idx + row_per_filter
        for i in range(row_start_idx,row_end_idx):
            shared_grad = shared_grad + grad_conv_k2[i,int(start_idx_list[i]):(int(start_idx_list[i])+l)]
        for i in range(row_start_idx,row_end_idx):
            conv_mat_k[i,int(start_idx_list[i]):(int(start_idx_list[i])+l)] = \
            conv_mat_k[i,int(start_idx_list[i]):(int(start_idx_list[i])+l)] + shared_grad
#    conv_mat_k = conv_mat + grad_conv_k2
    return conv_mat_k, mlp1_w_k

def model_fit1(x_train,y_train,x_test,y_test,kernal_size,filter_num,eta,epochs,n_class,batch_size,
               data_shuffle,acfun,rseed,conv2d_w_factor,mlp1_w_factor):
    np.random.seed(rseed)
    r_field = (x_train[0,:,:].shape[0]-kernal_size[0]+1, x_train[0,:,:].shape[1]-kernal_size[1]+1)
    history = np.zeros((epochs,4))
    filter_w = np.random.random([kernal_size[0],kernal_size[1],filter_num])*conv2d_w_factor
    mlp1_w = np.random.random((n_class,r_field[0]*r_field[1]*filter_num))*mlp1_w_factor
    
    train_idx = np.array(list(range(x_train.shape[0])))
    conv_mat,start_idx_list,l,conv_mat_bool = get_conv_matrix(filter_w,x_train)
    conv_mat_k = conv_mat.copy()
    mlp1_w_k = mlp1_w.copy()
    for i in range(epochs):
        print('Epoch '+str(i))
        if data_shuffle:
            np.random.shuffle(train_idx)
        num_step = np.ceil(x_train.shape[0]/batch_size)
        for n in range(int(num_step)):
#            print('step '+str(n+1)+'/'+str(int(num_step)))
            batch_idx = train_idx[range(n*batch_size,min((n+1)*batch_size,x_train.shape[0]))]
            x = x_train[batch_idx,:,:]
            y_true = y_train[batch_idx,:]
            conv_mat_k,mlp1_w_k = back_prop1(x,y_true,conv_mat_k,mlp1_w_k,eta,acfun,
                                             start_idx_list,l,conv_mat_bool,filter_num)

        y_pred = predict1(x_train,conv_mat_k,mlp1_w_k,acfun)
        y_pred_val = predict1(x_test,conv_mat_k,mlp1_w_k,acfun)
        history[i,0] = get_loss(y_train,y_pred)
        history[i,1] = get_loss(y_test,y_pred_val)
        history[i,2] = get_acc(y_train,y_pred)
        history[i,3] = get_acc(y_test,y_pred_val)
        print('Epoch '+str(i)+' Loss: '+str(round(history[i,0],4))+' Val loss: '+str(round(history[i,1],4))\
              +' Acc: '+str(round(history[i,2],4))+' Val acc: '+str(round(history[i,3],4)))
    return history,conv_mat_k,mlp1_w_k


def predict2(x,conv_mat,mlp1_w,mlp2_w,acfun):
    x_flatten = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])).transpose()
    conv2d_v = np.matmul(conv_mat,x_flatten)
    conv2d_y = relu(conv2d_v)
    mlp1_v = np.matmul(mlp1_w,conv2d_y)
    mlp1_y = relu(mlp1_v)
    mlp2_v = np.matmul(mlp2_w,mlp1_y)
    output = acfun(mlp2_v)
    output = output.transpose()
    return output
    


def back_prop2(x,y_true,conv_mat,mlp1_w,mlp2_w,eta,acfun,start_idx_list,l,conv_mat_bool,filter_num):
    x_flatten = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
    conv2d_v = np.matmul(conv_mat,x_flatten.transpose())
    conv2d_y = relu(conv2d_v)
    mlp1_v = np.matmul(mlp1_w,conv2d_y)
    mlp1_y = relu(mlp1_v)
    mlp2_v = np.matmul(mlp2_w,mlp1_y)
    output = acfun(mlp2_v)
    output = output.transpose()
    
    # update output weights:
    err_H_k = y_true-output
    d_H_k = (-1)* np.multiply(err_H_k ,acfun(mlp2_v.transpose(),True))
    d_H_k1 = np.reshape(d_H_k,(d_H_k.shape[0],d_H_k.shape[1],1))
    grad_mlp2_w_k = np.multiply(d_H_k1,np.reshape(mlp1_y.transpose(),
                                      (mlp1_y.shape[1],1,mlp1_y.shape[0])))
    grad_mlp2_w_k1 = (-eta)*np.mean(grad_mlp2_w_k,axis = 0)
    mlp2_w_k = mlp2_w + grad_mlp2_w_k1
    
    d_mlp1_k = np.multiply(relu(mlp1_v.transpose(),True),np.matmul(d_H_k,mlp2_w))
    d_mlp1_k1 = np.reshape(d_mlp1_k,(d_mlp1_k.shape[0],d_mlp1_k.shape[1],1))
    grad_mlp1_w_k = np.multiply(d_mlp1_k1,np.reshape(conv2d_y.transpose(),
                                      (conv2d_y.shape[1],1,conv2d_y.shape[0])))
    grad_mlp1_w_k1 = (-eta)*np.mean(grad_mlp1_w_k,axis = 0)
    mlp1_w_k = mlp1_w + grad_mlp1_w_k1    
    
    # update convolution weights:
    d_conv_k = np.multiply(np.matmul(d_mlp1_k,mlp1_w),relu(conv2d_v.transpose(),True))
    d_conv_k1 = np.reshape(d_conv_k,(d_conv_k.shape[0],d_conv_k.shape[1],1))
    grad_conv_k = np.matmul(d_conv_k1, np.reshape(x_flatten,(x_flatten.shape[0],1,x_flatten.shape[1])))
    grad_conv_k1 = (-eta)*np.mean(grad_conv_k,axis = 0)
    grad_conv_k2 = grad_conv_k1 * conv_mat_bool
    
    conv_mat_k = conv_mat.copy()
    row_per_filter = int(grad_conv_k1.shape[0]/filter_num)
    for filter_idx in range(filter_num):
        shared_grad = np.zeros(l)
        row_start_idx = filter_idx*row_per_filter
        row_end_idx = row_start_idx + row_per_filter
        for i in range(row_start_idx,row_end_idx):
            shared_grad = shared_grad + grad_conv_k2[i,int(start_idx_list[i]):(int(start_idx_list[i])+l)]
        for i in range(row_start_idx,row_end_idx):
            conv_mat_k[i,int(start_idx_list[i]):(int(start_idx_list[i])+l)] = \
            conv_mat_k[i,int(start_idx_list[i]):(int(start_idx_list[i])+l)] + shared_grad
#    conv_mat_k = conv_mat + grad_conv_k1
    return conv_mat_k, mlp1_w_k, mlp2_w_k

def model_fit2(x_train,y_train,x_test,y_test,kernal_size,filter_num,eta,epochs,n_class,batch_size,
               data_shuffle,acfun,rseed,conv2d_w_factor,mlp1_w_factor,mlp2_w_factor):
    np.random.seed(rseed)
    r_field = (x_train[0,:,:].shape[0]-kernal_size[0]+1, x_train[0,:,:].shape[1]-kernal_size[1]+1)
    
    filter_w = np.random.random([kernal_size[0],kernal_size[1],filter_num])*conv2d_w_factor
    mlp1_w = np.random.random((128,r_field[0]*r_field[1]*filter_num))*mlp1_w_factor
    mlp2_w = np.random.random((n_class,128))*mlp2_w_factor
    conv_mat,start_idx_list,l,conv_mat_bool = get_conv_matrix(filter_w,x_train)

    history = np.zeros((epochs,4))
    train_idx = np.array(list(range(x_train.shape[0])))
    for i in range(epochs):
        print('Epoch '+str(i))
        if data_shuffle:
            np.random.shuffle(train_idx)
        num_step = np.ceil(x_train.shape[0]/batch_size)
        for n in range(int(num_step)):
#            print('step '+str(n+1)+'/'+str(int(num_step)))
            batch_idx = train_idx[range(n*batch_size,min((n+1)*batch_size,x_train.shape[0]))]
            x = x_train[batch_idx,:,:]
            y_true = y_train[batch_idx,:]
            conv_mat_k,mlp1_w_k,mlp2_w_k = back_prop2(x,y_true,conv_mat,
                                                          mlp1_w,mlp2_w,eta,acfun,start_idx_list,l,
                                                          conv_mat_bool,filter_num)
            conv_mat = conv_mat_k.copy()
            mlp1_w = mlp1_w_k.copy()
            mlp2_w = mlp2_w_k.copy()
        y_pred = predict2(x_train,conv_mat,mlp1_w,mlp2_w,acfun)
        y_pred_val = predict2(x_test,conv_mat,mlp1_w,mlp2_w,acfun)
        history[i,0] = get_loss(y_train,y_pred)
        history[i,1] = get_loss(y_test,y_pred_val)
        history[i,2] = get_acc(y_train,y_pred)
        history[i,3] = get_acc(y_test,y_pred_val)
        print('Epoch '+str(i)+' Loss: '+str(round(history[i,0],4))+' Val loss: '+str(round(history[i,1],4))+
              ' Acc: '+str(round(history[i,2],4))+' Val acc: '+str(round(history[i,3],4)))
    return history,conv_mat_k,mlp1_w_k,mlp2_w_k