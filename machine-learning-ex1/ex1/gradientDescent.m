function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

currentTheta = theta;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    summation0=0;
    for i=1:m,
        summation0 =  summation0 + ((currentTheta(1)+(currentTheta(2)*X(i,2))) - y(i)) .* X(i,1);
    end 

    summation1=0;
    for i=1:m,
        summation1 = summation1 + ((currentTheta(1)+(currentTheta(2)*X(i,2))) - y(i)) .* X(i,2);
    end

theta(1) = currentTheta(1) - ((alpha.*summation0)/m);
theta(2) = currentTheta(2) - ((alpha.*summation1)/m);
currentTheta = theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
arr = [];
for i=1:m,
    arr(i)=(theta(2)*X(i,2)) + theta(1);
end
plotData(X, y);
hold on;
plot(arr);

end
