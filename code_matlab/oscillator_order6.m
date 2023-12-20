clear all;

A = [0 -1 2; 1 0 3; -2 -3 0];


% % Butcher tableau
npts = 2;
nder = 4;
RKA  = generate_HBRK_tables([0 : 1/(npts-1) : 1], nder*ones(1,npts));
format rat
for ii = 1 : nder
    RKA{ii}
end
pause
stages = size(RKA{1},1);
ee     = ones(stages,1);

syms a b c as bs cs;

assume(a, "real");
assume(b, "real");
assume(c, "real");
assume(as, "real");
assume(bs, "real");
assume(cs, "real");

y = [a; b; c]; ys = [as; bs; cs];

% Definition of y^{n+1}
syms dt; assume(dt, "real")
n = size(A, 1);
mat = eye(stages*n);
for ii = 1 : nder
    mat = mat - dt^ii * kron(RKA{ii},A^ii);
end
ys = mat^(-1) * kron(ee, y);

yn1 = ys(end-n+1:end)

%res_solution=compute_ode_solution(@(t,y) A*y, [1;2;3], dt);
%double(norm(res_solution - yn1))

simplify(yn1' * yn1)




