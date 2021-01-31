function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
X = [ones(length(X),1) X]
%Theta1
Theta2
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

store=zeros(length(p),length(Theta1));
for i=1:length(Theta1),
    for j=1:length(X),
        store(j,i)=sigmoid(X(j,:) * Theta1(i,:)');
    end
end
store;
store=[ones(length(store),1) store];
zeros(length(store),length(Theta2));
[rowSize,colSize] = size(Theta2);
storeLatest=zeros(rowSize,colSize);
for i=1:rowSize,
   for j=1:length(store),
      storeLatest(j,i)=sigmoid(store(j,:) * Theta2(i,:)');
    end
end
[val,index] = max(storeLatest, [], 2);
p=index;



% =========================================================================


end
