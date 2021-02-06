function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
Theta1;
Theta2;
X = [ones(m,1) X];
%size(X)
%y
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Calculate for each label
%activationSum = zeros(num_labels,1);
%zeros(size(X, 1), 1)
%for i=1:num_labels,
a1 = X
a2 = 0;
a3=0;
% First hidden Layer
layer1RegularizedValue = 0;
store=zeros(length(X),length(Theta1));
for i=1:length(Theta1),
    for j=1:length(X),
        store(j,i)=sigmoid(Theta1(i,:) * X(j,:)');
    end
end
a2 = store

[rowSize,colSize] = size(Theta1);
for i=1:rowSize,
    for j=2:colSize,
        layer1RegularizedValue=layer1RegularizedValue+(Theta1(i,j)^2);
    end
end

% Second hidden Layer
store=[ones(length(store),1) store];
zeros(length(store),length(Theta2));
[rowSize,colSize] = size(Theta2);
storeLatest=zeros(m,rowSize);
for i=1:rowSize,
   for j=1:length(store),
      storeLatest(j,i)=sigmoid(Theta2(i,:) * store(j,:)');
    end
end
a3 = storeLatest

layer2RegularizedValue = 0;

[rowSize,colSize] = size(Theta2);
for i=1:rowSize,
    for j=2:colSize,
        layer2RegularizedValue=layer2RegularizedValue+(Theta2(i,j)^2);
    end
end

% Matrix that shows the label bias. i.e. 1 for the current label, 0 for others 
innerY = zeros(length(y), num_labels);
for i=1:length(y),
    for j=1:num_labels,
        if (y(i)==j),
            innerY(i,j) = 1;
        end
    end 
end

% Cost function J calculation
summation = 0;
for i=1:m,
    innerSum = 0;
    for j=1:num_labels,
        v= innerY(i,j);
        innerSum = innerSum + (-1*v*log(storeLatest(i,j))) - ((1-v)*(log(1-storeLatest(i,j))));
    end
    summation = summation + innerSum;
end
J=summation/m;
J=J+((lambda * (layer2RegularizedValue+layer1RegularizedValue))/(2*m));

%diff3=zeros()
for i = 1:m,

end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
