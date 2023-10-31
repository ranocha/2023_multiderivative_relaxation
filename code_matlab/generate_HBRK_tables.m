% This functions computes the continuous Runge-Kutta tableaux that 
% correspond to a Hermite-Birkhoff quadrature formula. 
% 
% Multiple derivatives -- with the number of derivatives possibly variying
% from stage to stage -- can be taken into account. 
%
% The procedure is as follows: 
% Based on given coefficients 0 <= c_i <= 1, (c_i ~= c_j, i ~= j), the
% stage values are approximations to w(t^n + c_i dt). The final update
% w^{n+1} is computed as 
%
%     w^{n+1} = w^n + \int_{t^n}^{t^{n+1}} Pf(t) dt.  
%
% Pf is a polynomial of degree \sum_{numDer_i} - 1, where numDer_i denotes
% the number of derivatives that should be taken into account at stage i.
% The function PF is built in a way that 
% 
% Pf(t^n + c_i dt)^{(k)} = f^{(k)}(w^{n,i}), 0 <= k <= numDer_i - 1
% 
% The stages values are then again determined as 
% 
% w^{n,i} = w^n + \int_{t^n}^{t^n+c_i dt} Pf(t) dt, 
% 
% so in an implicit manner. This explains the fully implicit character of
% the method. 
%
% Examples: 
% generate_HBRK_tables([0 1/2 1], [1 1 1])        : Third order Lobatto IIIA
% generate_HBRK_tables([ 1/3 1], [1 1])           : Third order Radau IIA
% generate_HBRK_tables([ 0 1]        , [1 2])     : HB-I2DRK3-2s
% generate_HBRK_tables([ 0 1]        , [2 2])     : HB-I2DRK4-2s
% generate_HBRK_tables([ 0 1/2 1]    , [2 2 2])   : HB-I2DRK6-3s
% generate_HBRK_tables([ 0 1]        , [3 3])     : HB-I3DRK6-2s
% generate_HBRK_tables([ 0 1]        , [4 4])     : HB-I4DRK8-2s
% generate_HBRK_tables([ 0 1/3 2/3 1], [2 2 2 2]) : Eq. (3) from [1]
% [1] = Schütz, Seal, Zeifang, Parallel-in-time high-order multiderivative
% IMEX solvers, Journal of Scientific Computing, 2021

% 
% Author: Jochen Schütz
% Date  : 2021/12/08
% 
% Input arguments: 
% c        : the vector containing the time instances at the stages. There
%            should hold that 0 <= c_i <= 1, but extrapolation is assumed
%            to work as well, probably less stable. There must hold c_i ~=
%            c_j for i ~= j. 
% numDer   : How many derivatives do we want to take into account at stage
%            i? 
%
% Output arguments:
% The Butcher tableaux RKA, RKb, RKc. 
% Please note: If one is not interested in the continuous variant, then in
% RKb, simply set theta to one. 


function [RKA, RKb, RKc] = generate_HBRK_tables(c, numDer)

  if (length(c) ~= length(unique(c)))
      error ("Please supply a vector c with pairwise distinct coefficients!")
  end
  if (any(c>1) || any(c<0))
      warning("It is advised to use 0 <= c <= 1!")
  end
  if (length(c) ~= length(numDer))
      error("Number of integration points and derivatives must coincide!")
  end
  if (any(ceil(numDer) ~= numDer) || any(numDer <= 0))
      error("numDer must be a vector of positive integers")
  end
  
  % The first output argument is trivial to obtain. 
  RKc = c;
  
  % pp is the order of the polynomial that is built up for interpolating f.
  Ni = length(c);
  pp = sum(numDer) - 1; 
  
  % We need to have a basis, equivalent to the Lagrange basis polynomials.
  % For the sake of extendability, we make this numerically here. This
  % could come at the price of reduced stability in computing the Butcher
  % tableaux. If a problem comes up, it might be because of this, in
  % particular for very high orders. 
  A = [];
  syms x;
  % Maybe chose another basis here at some point. A Newton-basis seems the 
  % best choice to me at this moment -- not implemented yet! 
  ff = x.^[0:pp];
  for ii = 1 : Ni
      for kk = 1 : numDer(ii)
          A = [A; double(subs(diff(ff, kk-1), x, c(ii)))];
      end
  end
  if (cond(A) > 1e10)
      warning("This is a large condition number...")
  end
  % The j-th row in this matrix gives the coefficients of the j-th basis 
  % polynomial. 
  coeff_basis_polyn = inv(A);
  
  % Save the basis polynomials. We will add 'zero-'basis functions if there
  % is no additional derivative. This facilitates setting up the Butcher
  % matrices. 
  max_numDer = max(numDer);
  cnt = 1;
  for ii = 1 : Ni
      for kk = 1 : max_numDer
          l{ii, kk} = 0 * x;
          if (kk <= numDer(ii)) 
            for qq = 1 : pp + 1
              l{ii, kk} = l{ii, kk} + coeff_basis_polyn(qq, cnt) * ff(qq); 
            end
            cnt = cnt + 1;
          end
      end
  end
  
  % In the sequel, the Butcher matrices are computed through an integral.   
  for kk = 1 : max_numDer
      RKA{kk} = [];
  end
  for i0 = 1 : Ni
      for kk = 1 : max_numDer
          res{kk} = [];
      end
      for ii = 1 : Ni
        for kk = 1  : max_numDer
          res{kk} = [res{kk}, int(l{ii, kk}, 0, c(i0))];
        end
      end
      for kk = 1 : max_numDer
          RKA{kk} = [RKA{kk}; double(res{kk})];
      end
  end
  
  syms theta;
  for kk = 1 : max_numDer
      RKb{kk} = [];
  end
  for ii = 1 : Ni
      for kk = 1  : max_numDer
          RKb{kk} = [RKb{kk}, (int(l{ii, kk}, 0, theta))];
      end
  end
  
  
end
