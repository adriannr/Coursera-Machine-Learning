function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
%fprintf('%d %d %d %d \n', z(1), z(2), z(3), size(z,2));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i=1:size(z,1)
    for j=1:size(z,2)
        g(i,j)=1/(1+(exp(1)^(-z(i,j))));
    end
end





% =============================================================

end