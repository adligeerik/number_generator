clear all
close all


x = linspace(-4,4,1000);
relu = (x>0).*x;
lrelu = (x<0).*x.*0.1+(x>0).*x;
sigmoid = (1+exp(-x)).^(-1);
figure(1)

plot(x,tanh(x))
hold on;
plot(x,relu)
plot(x,lrelu)
plot(x,sigmoid)
xlim([-4 4])
ylim([-1.3 1.3])
ax = gca;
ax.YAxisLocation = 'origin';
%ax.XAxisLocation = 'origin';
ax.Box = 'off';
ax.Layer = 'top';
legend('tanh','ReLU','LeakyReLU','sigmoid','Location','southeast') 

