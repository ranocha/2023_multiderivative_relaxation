clear all;

syms y1 y2;
syms z1 z2;

f (y1,y2) = [-y2; y1] / (y1^2 + y2^2);
df(y1,y2) = jacobian(f(y1,y2));

dotf (y1,y2) = df(y1,y2) * f(y1,y2);
ddotf(y1,y2) = jacobian(dotf(y1,y2));

dt = 0.1;
y1 = 1; y2 = 0;

rhs = [y1;y2] + dt/2 * f(y1,y2) + dt^2/12 * dotf(y1,y2);
res = solve([z1; z2] - dt/2 * f(z1,z2) + dt^2/12 * dotf(z1,z2) == rhs);

