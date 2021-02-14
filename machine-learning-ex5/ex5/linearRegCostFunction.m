function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

lamdaSum = 0;
for j=2:size(theta),
    lamdaSum = lamdaSum + (theta(j)^2);
end

for i = 1:size(theta),
sum = 0;
gradientSum = 0;
    for j=1:m,
        sum = sum + (((X(j,:)*theta) - y(j))^2);
        gradientSum = gradientSum + (((X(j,:)*theta) - y(j))*(X(j,i)));
    end
    if (i == 1),
        grad(i) = (gradientSum/m);
    else,
        grad(i) = (gradientSum/m) + ((theta(i)*lambda)/m);
    end
end
J = (sum/(2*m)) + ((lamdaSum*lambda)/(2*m));
grad;


% =========================================================================

grad = grad(:);

end
