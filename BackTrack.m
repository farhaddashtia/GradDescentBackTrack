function [t] = BackTrack(A,x,alpha,beta)
t = 1;
fk=-sum(log(1-A*x))-sum(log(1-x.*x));
grad_f=A'*(1./(1-A*x))+ 2*x./(1-x.*x);
d=-grad_f;
x_first = x;

while ((max(A*(x+t*d)) >= 1) || (max(abs(x+t*d)) >= 1)),
t = beta*t;
end;

x = x +t*d;
fk1=-sum(log(1-A*x))-sum(log(1-x.*x));
while fk1 >= fk + alpha*t*(grad_f'*d)
  t = beta*t;
  x = x_first + t*d;
  fk1=-sum(log(1-A*x))-sum(log(1-x.*x));
end

end

