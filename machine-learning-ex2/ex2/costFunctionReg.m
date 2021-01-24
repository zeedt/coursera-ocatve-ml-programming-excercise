function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


sumLamda = 0;
for i=2:length(theta);
    sumLamda = sumLamda + (theta(i)^2);
end

sumForCost = 0;
for i=1:m,
    sigmoidValue = sigmoid(X(i,:) * theta);
    sumForCost = sumForCost + ((((-1*y(i))*log(sigmoidValue)) - ((1-y(i)) * log(1-sigmoidValue))));
end
J = (sumForCost/m)+((lambda*sumLamda)/(2*m));


for i=1:length(theta),
    sum = 0;
    for j=1:m,
        sigmoidValue = sigmoid(X(j,:) * theta);
        sum = sum + ((sigmoidValue-y(j))*X(j,i));
    end
    if i == 1,
        grad(i) = (sum/m);
    else,
        grad(i) = (sum/m) + ((lambda*theta(i))/m);
end
% =============================================================

end
