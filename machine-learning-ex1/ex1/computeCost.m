function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


for i=1:m,
    J = J + (((X(i,:) * theta) - y(i))^2);
end
J = J/(2*m);
%J=sprintf("%.2f", J);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

arr = [];
for i=1:m,
    arr(i)=(theta(2)*X(i,2)) + theta(1);
end
plotData(X, y);
hold on;
plot(arr);



% =========================================================================

end
