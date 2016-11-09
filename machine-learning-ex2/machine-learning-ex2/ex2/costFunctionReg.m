function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
z=X*theta;
siz= size(z);
% You need to return the following variables correctly 
J = 0;



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J=(1/m)*((-y)'*log(sigmoid(z))-((ones(size(y))-y)'*log(ones(siz)-sigmoid(z)))) + (lambda/(2*m))*sum(theta(2:end).^2) ;

grad= (1/m)*((sigmoid(z)-y)'*X) + ((lambda/m)*theta .* [0; ones(length(theta)-1, 1)])';

fprintf(' %d \n', size(grad));


% =============================================================

end
