clear all

syms dt A;

% % Butcher tableau
nder = 5;
RKA  = generate_HBRK_tables([0 : 1/2 : 1], [nder, nder, nder]);

for ii = 1 : nder
    RKA{ii} = double(RKA{ii})
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Necessary definitions
B21 = dt * RKA{1}(2,1) * A;
B22 = dt * RKA{1}(2,2) * A;
B23 = dt * RKA{1}(2,3) * A;
B31 = dt * RKA{1}(3,1) * A;
B32 = dt * RKA{1}(3,2) * A;
B33 = dt * RKA{1}(3,3) * A;

B21T = -dt * RKA{1}(2,1) * A;
B22T = -dt * RKA{1}(2,2) * A;
B23T = -dt * RKA{1}(2,3) * A;
B31T = -dt * RKA{1}(3,1) * A;
B32T = -dt * RKA{1}(3,2) * A;
B33T = -dt * RKA{1}(3,3) * A;


for ii = 2 : nder
    B21 = B21 + dt^ii * RKA{ii}(2,1) * A^ii;
    B22 = B22 + dt^ii * RKA{ii}(2,2) * A^ii;
    B23 = B23 + dt^ii * RKA{ii}(2,3) * A^ii;
    B31 = B31 + dt^ii * RKA{ii}(3,1) * A^ii;
    B32 = B32 + dt^ii * RKA{ii}(3,2) * A^ii;
    B33 = B33 + dt^ii * RKA{ii}(3,3) * A^ii;

    B21T = B21T + (-1)^ii * dt^ii * RKA{ii}(2,1) * A^ii;
    B22T = B22T + (-1)^ii * dt^ii * RKA{ii}(2,2) * A^ii;
    B23T = B23T + (-1)^ii * dt^ii * RKA{ii}(2,3) * A^ii;
    B31T = B31T + (-1)^ii * dt^ii * RKA{ii}(3,1) * A^ii;
    B32T = B32T + (-1)^ii * dt^ii * RKA{ii}(3,2) * A^ii;
    B33T = B33T + (-1)^ii * dt^ii * RKA{ii}(3,3) * A^ii;    
end

QS =1 - B22;
QST=1 - B22T;


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End definitions


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start formlulas

% Actually, AA and BB are the definitions multiplied by S !! 
AA = (-B32 * B23 + QS - QS*B33);
BB = (QS + QS*B31 + B32 * (1+B21));

% AA * AA' - BB * BB' = 0 ? 
tmp = (-B32 * B23 + QS - QS*B33) * (-B32T * B23T + QST - QST*B33T);
tmp = tmp - (QS + QS*B31 + B32 * (1+B21)) * (QST + QST*B31T + B32T * (1+B21T));

simplify(tmp)
