clc
clear all
%%
n=50;
m=100;
alpha=0.01;
beta=0.5;
eta=0.001;
A=zeros(m,n);
for i=1:m
    A(i,:)=rand(1,n)-0.5;
end
x=zeros(n,1);
j=0; 
grad_f=A'*(1./(1-A*x))+ 2*x./(1-x.*x);
f=zeros(1000,1);
t=zeros(1000,1);
max_iteration=1000;

while j < max_iteration
    j=j+1;
    grad_f=A'*(1./(1-A*x))+ 2*x./(1-x.*x);
    delta_x=-grad_f;
    t(j)=BackTrack(A,x,alpha,beta);
    x=x+t(j)*delta_x;
    f(j)=-sum(log(1-A*x))-sum(log(1-x.*x));
    if norm(grad_f) <= eta
        break;
    end
end

figure(1)
plot(1:j,f(1:j),'lineWidth',2)
xlabel('iteration')
ylabel('Value of Objective Function')
grid on
figure(2)
stem(1:j,t(1:j),'lineWidth',1)
xlabel('iteration')
ylabel('Step Length')
grid on
figure(3)
plot(1:j,f(1:j)-f(j),'lineWidth',2)
xlabel('iteration')
ylabel('f - p^*')
grid on
