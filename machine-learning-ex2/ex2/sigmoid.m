function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
[rowSize, colSize] = size(z);
g = zeros(size(z));
for i=1:rowSize,
    for j=1:colSize,
        g(i,j) = 1 / (1+(e^(-1*(z(i,j)))));
    end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
