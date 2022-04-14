function p = zerosum( A )
%ZEROSUM solves a matrix zero sum game 
% Input:
%    A - game matrix - square matrix. gain for maximizer's in rows.
% Output:
%    p - distribution of the maximizer's choice. 
% Created: Kaiser Asif @UIC. 11/22/2014
% ref: 4.4 of http://www.math.ucla.edu/~tom/Game_Theory/mat.pdf


f = ones(size(A, 1), 1); % function should only sum x_i.  

A = -A; % theory [1] <= xA. Matlab linprog Ax <= b
b = -f; % [-1]. "[1] <= xA." to "Ax <= b"
lb = zeros(size(f)); % all x >= 0

A

[x,fval,exitflag,output,lambda] = linprog(f,A,b,[],[],lb);

x

v = 1 / sum(x); % see ref. 

p = x * v; % see ref

end

