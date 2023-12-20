function res = compute_ode_solution(rhs, initial, tt)

      if (tt ~= 0) 
        warning('off', 'all')
        options = odeset('RelTol',1e-16,'AbsTol',1e-16*ones(length(rhs),1));
        [~,Y] = ode15s(rhs, [0 tt], initial, options);
        warning('on', 'all')
        res = Y(end,:)'; 
      else
          res = initial;
      end
      
      
end