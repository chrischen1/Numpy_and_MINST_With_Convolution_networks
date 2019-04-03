
# Numpy and MNIST with convolution 
Based on the submission of Assignment1 of MU EECS CMP_SC 8770(2019Spring)

This is a simple demo for convolution neural networks with Numpy only on Python. The sample inputs are from MINST.

The following procedures are all based on matrix multiplication to ensure minimum use of for loops in python except for the update of conv2d_mat from grad_conv_k1. However, the code for both matrix multiplication version and loop version are provided.

## Forward pass computations:

1. The training data are 28*28 images from MINST datasets, are flattened to vector x_flatten(n,784)  where n is the size all samples used in this step. 

2. The first step is always computing conv2d_mat(r,784) × T(x_flatten)(784,n)  in order to compute convolution output conv2d_v with shape (r,n).

3. Apply RELU on conv2d_v (r,n) to get conv2d_y (r,n).

4. Dense layers

    a) If next layer is the final layer, mlp1_w would be a matrix of (d,r), where d is the dimension of output layer. By computing mlp1_w(d,r) × conv2d_y(r,n) we get mlp1_v with shape (d,n) and apply activation function used for last layer and transpose the results to get output of current batch output(n,d).

    b) If next layer is followed by a hidden layer of size h, mlp1_w would be a matrix of (h,r), where h is the dimension of hidden layer(h is always set to 128 in this project). By computing mlp1_w(h,r) × conv2d_y(r,n) we get mlp1_v(h,n) and apply RELU on it to get mlp1_y(h,n). A second weight matrix mlp2_w(d,h) is used to get final output where d is the dimension of the output layer. By computing mlp2_w(d,h) ×mlp1_y(h,n) we get mlp2_v(d,n) and activation function used for last layer and transpose the results to get output of current batch output (n,d).

## Backward pass computations (* represents element-wise multiplications):

1. The loss is calculated as MSE, and training error err_H_k(n,d)  is computed as
 y_{true(n,d)}-y_{pred}(n,d)  where n is the size of current batch(can be 1 or all training data depending on the data presentation).
Depending on the number of MLP layers, the procedures to get gradients are:
2. Backprop on Dense layers.

    a)If there is only one MLP layer: ƍH = d_H_k(n,d)  = (-1) * err_H_k(n,d)  * f'(T(mlp1_v)) (n,d). 
    
    Then reshape d_H_k(n,d)  by row major to get d_H_k1(n,d,1), reshape T(conv2d_y) (n,r) by row major to C (n,1,r) and computing d_H_k1(n,d,1) × C(n,1,r)  for the last two dimension and broadcasting on the first dimension to get grad_mlp1_w_k (n,d,r). 
 
    Average on the first dimension and multiply the learning rate eta we get grad_mlp1_w_k1(d,r) which is the value to update mlp1_w(d,r)
 
    b) If there are 2 MLP layers:
    
    First compute ƍH = d_H_k(n,d)  = (-1) * err_H_k(h,d)  * f'(T(mlp2_v) (n,d). Then reshape d_H_k(n,d) by row major to get d_H_k1(n,d,1), reshape T(mlp2_y) (n,h) by row major to C1 (n,1,h) and computing d_H_k1(n,d,1) × C(n,1,h)  on the last two dimension and broadcasting on the first dimension to get grad_mlp2_w_k (n,d,h). 

    Average on the first dimension and multiply the learning rate eta we get grad_mlp2_w_k1(d,h) which is the value to update mlp2_w(d,h)

    Then compute ƍH-1n,h=d_mlp1_k =f'(T(mlp1_v)) (n,h)) * (d_H_k(n,d) × mlp2_w(d,h))
    
    Reshape d_mlp1_k(n,h) to (n,h,1) by row major to get d_mlp1_k1(n,h,1), reshape T(conv2d_y) (n,r) by row major to C (n,1,r) and computing d_mlp1_k1(n,h,1)× C(n,1,r) )  on the last two dimension and broadcasting on the first dimension to get grad_mlp1_w_k (n,h,r).
    
    Average on the first dimension and multiply the learning rate eta we get grad_mlp1_w_k1(h,r) which is the value to update mlp1_w(h,r)

3. compute ƍconv = d_conv_k (n,r) = (d_mlp1_k(n,h) ×  mlp1_w(h,r) × f'(T(conv2d_v)) (n,r) )
4. Reshape d_conv_k (n,r) on row major to get d_conv_k1 (n,r,1) 
5. Reshape x_flatten (n,784)  on row major to get X(n,1,784)  
6. Compute grad_conv_k(n,r,784) = d_conv_k1 (n,r,1)  × X (n,1,784), do the same on the last two dimension and broadcasting on the first dimension.
7. Average on the first dimension and multiply the learning rate eta we get grad_conv_k1 (r,784) which is the value to update conv2d_mat(r,784)
8. For each row of conv2d_mat, since the weights are shared, we just accumulate the gradient from all rows for the same weight according to the start index of the weights in each row. This would condense the gradient to one row of length 784 and then we rebuild the conv2d_mat reversely. 

At the end of each epoch, the loss and accuracy of training and testing data are computed. The output with highest value is the predicted class and used for evaluation of accuracy.
