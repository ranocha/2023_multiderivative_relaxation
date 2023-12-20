clear all;

A = [0 -1 2; 1 0 3; -2 -3 0];


syms a b c as bs cs;

assume(a, "real");
assume(b, "real");
assume(c, "real");
assume(as, "real");
assume(bs, "real");
assume(cs, "real");

y = [a; b; c]; ys = [as; bs; cs]

(A*y + A*ys)' * (A^2*y-A^2 *ys)



% Definition of y^{n+1}
syms dt; assume(dt, "real")
n = size(A, 1);
ys = (eye(n) - dt/2 * A + dt^2/12 * A^2)^(-1) * (eye(n) + dt/2 * A + dt^2/12 * A^2) * y 

AA = (eye(n) - dt/2 * A + dt^2/12 * A^2);
BB = eye(n) + dt/2 * A + dt^2/12 * A^2;

ys' * ys
simplify(ans)