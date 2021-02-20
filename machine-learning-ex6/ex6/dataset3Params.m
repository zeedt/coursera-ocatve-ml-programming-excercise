function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.03;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%CArr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%SArr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

CArr = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8];
SArr = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8];
minError = 5000;
error = 0;
for i=1:length(CArr),
    for j=1:length(SArr),
        %model = svmTrain(Xval, yval, CArr(i), @(t)(gaussianKernel(Xval(:,1), Xval(:,2), SArr(j))));
        %predictions = svmPredict(model, Xval);
        %error = mean(double(predictions ~= yval));
        if (error < minError),
            minError = error;
            C = CArr(i);
            sigma = SArr(j);
        end
    end
end
minError
C
sigma


% =========================================================================

end
