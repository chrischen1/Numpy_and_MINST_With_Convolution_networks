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
    rf_size = np.array((1+input_mat.shape[0]-kernal_mat.shape[0],1+input_mat.shape[1]-kernal_mat.shape[1]))
    conv_mat = np.zeros((rf_size[0]*rf_size[1],input_mat.shape[1]*input_mat.shape[0]))
    row_base = np.array([])
    for i in range(kernal_mat.shape[0]):
        row_base = np.concatenate((row_base,kernal_mat[i,:],np.repeat(0,rf_size[1]-1)))
    row_base=row_base[0:(row_base.shape[0]-(input_mat.shape[1]-kernal_mat.shape[1]))]
    
    for i in range(rf_size[0]):
        for j in range(rf_size[1]):
            row_mat = np.zeros(conv_mat.shape[1])
            row_start_idx = i*input_mat.shape[1]+j
            row_mat[row_start_idx:(row_start_idx+row_base.shape[0])] = row_base
            conv_mat[j+i*rf_size[1],:] = row_mat
    return conv_mat        
            
def conv_transpose(kernal_mat,input_mat):
    conv_mat = get_conv_matrix(kernal_mat,input_mat)
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

def predict1(X,filter_w,mlp_w,acfun,n_class):
    kernal_size = filter_w.shape
    y_pred = np.zeros((X.shape[0],n_class))
    for idx in range(X.shape[0]):
        x = X[idx,:,:]
        r_field = (x.shape[0]-kernal_size[0]+1, x.shape[1]-kernal_size[1]+1)
        conv2d_v = np.zeros((r_field[0],r_field[1],filter_w.shape[2]))
        for k in range(filter_w.shape[2]):
            conv_mat = get_conv_matrix(filter_w[:,:,k],x)
    
            conv2d_v[:,:,k] = np.reshape(np.matmul(conv_mat,
                    np.reshape(x,(x.shape[0]*x.shape[1],1))),(r_field[0],r_field[1]))
        conv2d_y = relu(conv2d_v)
        conv2d_output = np.reshape(conv2d_y,(conv2d_y.shape[0]*conv2d_y.shape[1]*conv2d_y.shape[2],1))
        #MLP
        mlp_v = np.dot(np.reshape(mlp_w,(mlp_w.shape[0],conv2d_output.shape[0])),conv2d_output)
        output = acfun(mlp_v)
        y_pred[idx,:] = output[:,0]
    return y_pred

def back_prop1(x,y_true,filter_w,mlp_w,filter_w_k,mlp_w_k,eta,acfun):    
    #Conv2D
    kernal_size = filter_w.shape
    r_field = (x.shape[0]-kernal_size[0]+1, x.shape[1]-kernal_size[1]+1)
    conv2d_v = np.zeros((r_field[0],r_field[1],filter_w.shape[2]))
    for k in range(filter_w.shape[2]):
        conv_mat = get_conv_matrix(filter_w[:,:,k],x)

        conv2d_v[:,:,k] = np.reshape(np.matmul(conv_mat,
                np.reshape(x,(x.shape[0]*x.shape[1],1))),(r_field[0],r_field[1]))
    conv2d_y = relu(conv2d_v)
    conv2d_output = np.reshape(conv2d_y,(conv2d_y.shape[0]*conv2d_y.shape[1]*conv2d_y.shape[2],1))
    #MLP
    mlp_v = np.dot(np.reshape(mlp_w,(mlp_w.shape[0],conv2d_output.shape[0])),conv2d_output)
    output = acfun(mlp_v)
    # update output weights:
    err_H_k = y_true - output.flatten()
    d_H_k = np.zeros_like(err_H_k)
    for i in range(mlp_w.shape[0]):
        d_H_k[i] = (-1)* err_H_k[i] * acfun(mlp_v[i,0],True)
        for j in range(mlp_w.shape[1]):
            for k in range(mlp_w.shape[2]):
                for l in range(mlp_w.shape[3]):
                    mlp_w_k[i,j,k,l] += (-eta)*d_H_k[i]*conv2d_y[j,k,l]
    # update filter_w:
    d_conv_k = np.zeros((r_field[0],r_field[1],filter_w.shape[2]))
    for i in range(d_conv_k.shape[0]):
        for j in range(d_conv_k.shape[1]):
            for k in range(d_conv_k.shape[2]):
                for l in range(output.shape[0]):
                    d_conv_k[i,j,k] += d_H_k[l]*mlp_w[l,i,j,k]*relu(conv2d_v[i,j,k],True)
    for k in range(filter_w.shape[2]):
        for i in range(filter_w.shape[0]):
            for j in range(filter_w.shape[1]):
                for window_row in range(r_field[0]):
                    for window_col in range(r_field[1]):
                        filter_w_k[i,j,k] += (-eta)*d_conv_k[window_row,window_col,k]*x[i+window_row,j+window_col]
                        
    return filter_w_k,mlp_w_k

def model_fit1(x_train,y_train,x_test,y_test,kernal_size,filter_num,eta,epochs,n_class,batch_size,
               data_shuffle,acfun,rseed,conv2d_w_factor,mlp1_w_factor):
    np.random.seed(rseed)
    r_field = (x_train[0,:,:].shape[0]-kernal_size[0]+1, x_train[0,:,:].shape[1]-kernal_size[1]+1)

    filter_w = np.random.random([kernal_size[0],kernal_size[1],filter_num])*conv2d_w_factor
    mlp_w = np.random.random((n_class,r_field[0],r_field[1],filter_num))*mlp1_w_factor
    
    filter_w_k = filter_w.copy()
    mlp_w_k = mlp_w.copy()
    history = np.zeros((epochs,4))
    train_idx = np.array(list(range(x_train.shape[0])))
    for i in range(epochs):
        print('Epoch '+str(i))
        if data_shuffle:
            np.random.shuffle(train_idx)
        num_step = np.ceil(x_train.shape[0]/batch_size)
        for n in range(int(num_step)):
            batch_idx = train_idx[range(n*batch_size,min((n+1)*batch_size,x_train.shape[0]))]
#            print('step '+str(n+1)+'/'+str(int(num_step)))
            for s in batch_idx:
                filter_w_k,mlp_w_k = back_prop1(x_train[s,:,:],y_train[s,:],
                                               filter_w,mlp_w,filter_w_k,mlp_w_k,eta,acfun)
            filter_w = filter_w_k.copy()
            mlp_w = mlp_w_k.copy()
        y_pred = predict1(x_train,filter_w_k,mlp_w_k,acfun,n_class)
        y_pred_val = predict1(x_test,filter_w_k,mlp_w_k,acfun,n_class)
        history[i,0] = get_loss(y_train,y_pred)
        history[i,1] = get_loss(y_test,y_pred_val)
        history[i,2] = get_acc(y_train,y_pred)
        history[i,3] = get_acc(y_test,y_pred_val)
        print('Epoch '+str(i)+' Loss: '+str(round(history[i,0],4))+' Val loss: '+str(round(history[i,1],4))+
              ' Acc: '+str(round(history[i,2],4))+' Val acc: '+str(round(history[i,3],4)))
    return history,filter_w_k,mlp_w_k

def predict2(X,filter_w,mlp1_w,mlp2_w,acfun,n_class):
    y_pred = np.zeros((X.shape[0],n_class))
    kernal_size = filter_w.shape
    for idx in range(X.shape[0]):
        x = X[idx,:,:]
        r_field = (x.shape[0]-kernal_size[0]+1, x.shape[1]-kernal_size[1]+1)
        conv2d_v = np.zeros((r_field[0],r_field[1],filter_w.shape[2]))
        for k in range(filter_w.shape[2]):
            conv_mat = get_conv_matrix(filter_w[:,:,k],x)
    
            conv2d_v[:,:,k] = np.reshape(np.matmul(conv_mat,
                    np.reshape(x,(x.shape[0]*x.shape[1],1))),(r_field[0],r_field[1]))
        conv2d_y = relu(conv2d_v)
        conv2d_output = np.reshape(conv2d_y,(conv2d_y.shape[0]*conv2d_y.shape[1]*conv2d_y.shape[2],1))
        #MLP
        mlp_v1 = np.dot(np.reshape(mlp1_w,(mlp1_w.shape[0],conv2d_output.shape[0])),conv2d_output)
        mlp_y1 = relu(mlp_v1)
        mlp_v2 = np.dot(mlp2_w,mlp_y1)
        output = acfun(mlp_v2)
        y_pred[idx,:] = output[:,0]
    return y_pred

def back_prop2(x,y_true,filter_w,mlp1_w,mlp2_w,filter_w_k,mlp1_w_k,mlp2_w_k,eta,acfun):    
    #Conv2D
    kernal_size = filter_w.shape
    r_field = (x.shape[0]-kernal_size[0]+1, x.shape[1]-kernal_size[1]+1)
    conv2d_v = np.zeros((r_field[0],r_field[1],filter_w.shape[2]))
    for k in range(filter_w.shape[2]):
        conv_mat = get_conv_matrix(filter_w[:,:,k],x)

        conv2d_v[:,:,k] = np.reshape(np.matmul(conv_mat,
                np.reshape(x,(x.shape[0]*x.shape[1],1))),(r_field[0],r_field[1]))
    conv2d_y = relu(conv2d_v)
    conv2d_output = np.reshape(conv2d_y,(conv2d_y.shape[0]*conv2d_y.shape[1]*conv2d_y.shape[2],1))
    #MLP
    mlp_v1 = np.dot(np.reshape(mlp1_w,(mlp1_w.shape[0],conv2d_output.shape[0])),conv2d_output)
    mlp_y1 = relu(mlp_v1)
    mlp_v2 = np.dot(mlp2_w,mlp_y1)
    output = acfun(mlp_v2)
    # update output weights mlp2_w:
    err_H_k = y_true - output.flatten()
    d_H_k = np.zeros_like(err_H_k)
    for i in range(mlp2_w.shape[0]):
        d_H_k[i] = (-1)* err_H_k[i] * acfun(mlp_v2[i,0],True)
        for j in range(mlp2_w.shape[1]):
            mlp2_w_k[i,j] += (-eta)*d_H_k[i]*mlp_y1[j,0]
    # update output weights mlp1_w:
    d_mlp1_k = np.zeros(mlp1_w_k.shape[0])
    for i in range(mlp2_w.shape[1]):
        for j in range(mlp2_w.shape[0]):
            d_mlp1_k[i] += mlp2_w[j,i]* d_H_k[j] * relu(mlp_v1[i,0],True)
    for i in range(mlp1_w_k.shape[0]):        
        for j in range(mlp1_w_k.shape[1]):
            for k in range(mlp1_w_k.shape[2]):
                for l in range(mlp1_w_k.shape[3]):
                    mlp1_w_k[i,j,k,l] += (-eta)*d_mlp1_k[i]*conv2d_y[j,k,l]
    # update filter_w:
    d_conv_k = np.zeros((r_field[0],r_field[1],filter_w.shape[2]))
    for i in range(d_conv_k.shape[0]):
        for j in range(d_conv_k.shape[1]):
            for k in range(d_conv_k.shape[2]):
                for l in range(mlp1_w.shape[0]):
                    d_conv_k[i,j,k] += d_mlp1_k[l]*mlp1_w[l,i,j,k]*relu(conv2d_v[i,j,k],True)
    for k in range(filter_w.shape[2]):
        for i in range(filter_w.shape[0]):
            for j in range(filter_w.shape[1]):
                for window_row in range(r_field[0]):
                    for window_col in range(r_field[1]):
                        filter_w_k[i,j,k] += (-eta)*d_conv_k[window_row,window_col,k]*x[i+window_row,j+window_col]
    return filter_w_k,mlp1_w_k,mlp2_w_k

def model_fit2(x_train,y_train,x_test,y_test,kernal_size,filter_num,eta,epochs,n_class,batch_size,
               data_shuffle,acfun,rseed,conv2d_w_factor,mlp1_w_factor,mlp2_w_factor):
    np.random.seed(rseed)
    r_field = (x_train[0,:,:].shape[0]-kernal_size[0]+1, x_train[0,:,:].shape[1]-kernal_size[1]+1)
    
    filter_w = np.random.random([kernal_size[0],kernal_size[1],filter_num])*conv2d_w_factor
    mlp1_w = np.random.random((128,r_field[0],r_field[1],filter_num))*mlp1_w_factor
    mlp2_w = np.random.random((n_class,128))*mlp2_w_factor
    
    filter_w_k = filter_w.copy()
    mlp1_w_k = mlp1_w.copy()
    mlp2_w_k = mlp2_w.copy()
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
            for s in batch_idx:
                filter_w_k,mlp1_w_k,mlp2_w_k = back_prop2(x_train[s,:,:],y_train[s,:],
                                               filter_w,mlp1_w,mlp2_w,filter_w_k,mlp1_w_k,mlp2_w_k,eta,acfun)
            filter_w = filter_w_k.copy()
            mlp1_w = mlp1_w_k.copy()
            mlp2_w = mlp2_w_k.copy()
        y_pred = predict2(x_train,filter_w,mlp1_w,mlp2_w,acfun,n_class)
        y_pred_val = predict2(x_test,filter_w,mlp1_w,mlp2_w,acfun,n_class)
        history[i,0] = get_loss(y_train,y_pred)
        history[i,1] = get_loss(y_test,y_pred_val)
        history[i,2] = get_acc(y_train,y_pred)
        history[i,3] = get_acc(y_test,y_pred_val)
        print('Epoch '+str(i)+' Loss: '+str(round(history[i,0],4))+' Val loss: '+str(round(history[i,1],4))+
              ' Acc: '+str(round(history[i,2],4))+' Val acc: '+str(round(history[i,3],4)))
    return history,filter_w_k,mlp1_w_k,mlp2_w_k