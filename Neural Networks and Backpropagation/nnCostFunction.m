function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1 = [ones(m,1) X];
a2 = sigmoid(a1*Theta1');
a2 = [ones(m,1) a2];
hx = sigmoid(a2*Theta2');

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end


part1 = (Y).*log(hx);
part2 = (1-Y).*log(1-hx);


Theta1_reg = Theta1(:, 2:end).^2;
Theta2_reg = Theta2(:, 2:end).^2;
reg = (lambda/(2*m))*((sum(sum(Theta1_reg,2))) + (sum(sum(Theta2_reg,2))));

J = ((-1/m) * sum(sum(part1 + part2))) + reg;



% backpropagation algorithm
% cap_delta_1 = zeros(25,401);
% cap_delta_2 = zeros(26,10);
% 
% for t = 1:m
%     a1 = [ 1 X(t,:)];
%     a2 = sigmoid(a1*Theta1');
%     a2 = [ 1 a2];
%     a3 = sigmoid(a2*Theta2');
%     
%     for k = 1:num_labels
%         delta3(1,k) = (a3(1,k) - Y(1,k));
%     end
%     for k = 1:hidden_layer_size
%         p1 = Theta2(:,k)*delta3(1,k);
%         p2 = [1 (Theta1(:,k)*sigmoid(a1(:,k)))'];
%         delta2 = p1 .* p2;
%     end
%     
%     delta2 = delta2(2:end);
%     cap_delta_1 = cap_delta_1 + (a1'*delta2)';
%     cap_delta_2 = cap_delta_2 + (a2'*delta3);
% end
% 
% Theta1_grad = cap_delta_1/m;
% Theta2_grad = cap_delta_2/m;
% 

% corrected version of back propagation

delta1 = zeros(size(Theta1,1),size(Theta1,2));
delta2 = zeros(size(Theta2,1),size(Theta2,2));
for i=1:m
   bpa1 = [ 1 X(i,:)];
   bpa2 = sigmoid(bpa1*Theta1');
   bpa2 = [ 1 bpa2];
   bphx = sigmoid(bpa2*Theta2');
   
   d3 = bphx-Y(i,:);
   d2 = d3*Theta2;
   d2 = d2.*[1 sigmoidGradient(bpa1*Theta1')];
   d2 = d2(:,2:end);
   delta2 = delta2 + d3'*bpa2;
   delta1 = delta1 + d2'*bpa1;
   
end

Theta1_grad = delta1/m;
Theta2_grad = delta2/m;

Theta1_grad(:,2:end) = (delta1(:,2:end)/m + lambda/m*Theta1(:,2:end));
Theta2_grad(:,2:end) = (delta2(:,2:end)/m + lambda/m*Theta2(:,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
