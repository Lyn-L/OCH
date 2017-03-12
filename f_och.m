function f = f_och(W, X, SimMatrix)

Y = sign(X*W);

D = diag(sum(SimMatrix));

L = D - SimMatrix;

f = 0.5 * trace(Y' * L * Y);

end