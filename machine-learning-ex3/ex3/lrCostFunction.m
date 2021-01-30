function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



sumLamda = sum(theta(2:end).^2);
sumForCost=0;
sigmoidMatrix = sigmoid(X(:,:)*theta);
sumForCost = sum((-1*y.*log(sigmoidMatrix)) - ((1-y).*log(1-sigmoidMatrix))); % This vectorised code commented below
%for i=1:size(X,1),
%    sigmoidValue = sigmoid(X(i,:)*theta);
%    sumForCost = sumForCost + ((-1*y(i)*log(sigmoidValue)) - ((1-y(i)) * log(1-sigmoidValue)));
%end

J = (sumForCost/m)+((lambda*sumLamda)/(2*m));
for i=1:length(theta),
    sum = 0;
    for j=1:m,
        sigmoidValue = sigmoid(X(j,:) * theta);
        sum = sum + ((sigmoidValue - y(j))*X(j,i));
    end
    if i == 1,
        grad(i) = (sum/m);
    else,
        grad(i) = (sum/m) + ((lambda*theta(i))/m);
end






% =============================================================

grad = grad(:);

end