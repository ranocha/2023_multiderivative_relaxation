


%dt = rand;
%un = [100 * (rand - 0.5); 100 * (rand - 0.5)];

dt = 0.566468595993228/10;
un = [0.168433277116653; -0.401707557225028];


normm = @(x) x(1)^2 + x(2)^2;
f = @(x) [x(1) + dt / 2 / normm(x) * x(2); ... 
          x(2) - dt / 2 / normm(x) * x(1)];

df= @(x) [1 - dt/2 /normm(x)^2 * 2 * x(1) * x(2), dt/2/normm(x) - dt / 2 / normm(x)^2 * 2 * x(2)^2; ...
          -dt / 2 / normm(x) + dt / 2 / normm(x)^2 * 2 * x(1)^2, 1 - dt / 2 / normm(x)^2 * 2 * x(1) * x(2)];

f = @(x) f(x) - [un(1) - dt / 2 / normm(un) * un(2); ...
                 un(2) + dt / 2 / normm(un) * un(1)];

un1 = damped_newton(f, df, un, 100, 1e-16);

norm(f(un1),2)

res = un'*un-un1'*un1

if (abs(res) > 1e-10)
    dt
    un
end

