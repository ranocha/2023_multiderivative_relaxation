clear all;

A = [0 -1 2; 1 0 3; -2 -3 0];

%A = rand(4, 4);

A = A - A';


% % Butcher tableau
npts = 3;
nder = 2;
RKA  = generate_HBRK_tables([0 : 1/(npts-1) : 1], nder*ones(1,npts));
for ii = 1 : nder
    RKA{ii};
end


syms dt; assume(dt, "real")


B21 = dt * RKA{1}(2,1) * A + dt^2 * RKA{2}(2,1) * A^2;
B22 = dt * RKA{1}(2,2) * A + dt^2 * RKA{2}(2,2) * A^2;
B23 = dt * RKA{1}(2,3) * A + dt^2 * RKA{2}(2,3) * A^2;
B31 = dt * RKA{1}(3,1) * A + dt^2 * RKA{2}(3,1) * A^2;
B32 = dt * RKA{1}(3,2) * A + dt^2 * RKA{2}(3,2) * A^2;
B33 = dt * RKA{1}(3,3) * A + dt^2 * RKA{2}(3,3) * A^2;

Id = eye(size(A));
AA = -B32 * (Id-B22)^(-1) * B23 + (Id - B33);
BB = Id + B31 + B32 *(Id - B22)^(-1)*(Id + B21);
S  = (Id - B22)^(-1);

%subs(AA * AA', dt = 1/2)
%subs(BB * BB', dt = 1/2)

%simplify(AA * AA' - BB * BB')

