function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% list of C, sigma values to try
C_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

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

cv_error = [];

for i=1:8
  for j=1:8
    model = svmTrain(X, y, C_arr(i), @(x1, x2) gaussianKernel(x1, x2, sigma_arr(j)));
    predictions = svmPredict(model, Xval);
    cv_error(i,j) = mean(double(predictions ~= yval));
    fprintf('Run %d: C = %f sigma = %f, cv_error = %f \n', (i-1)*8+j, C_arr(i), sigma_arr(j), cv_error(i, j));
  end
end

[min_val,idx]=min(cv_error(:))
[row,col]=ind2sub(size(cv_error),idx)

C = C_arr(row);
sigma = sigma_arr(col);

% =========================================================================

end
