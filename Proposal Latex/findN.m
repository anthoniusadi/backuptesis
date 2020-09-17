function N=findN(lambda,mu,xi,v)

e2=10^(-6);

x1=(lambda+mu+xi-((lambda+mu+xi)^2-4*mu*xi)^0.5)/(2*lambda);
x2=(lambda+mu+xi+((lambda+mu+xi)^2-4*mu*xi)^0.5)/(2*lambda);

A=x2/(x2-x1);
B=x1/(x2-x1);

a=lambda*A/v;

rho=lambda*((1/mu) + (1/xi))
    
c0=((1-rho)/gamma(a)) * ((x1-1)/x1)^a * ((x2-x1)/(x2-1))^(lambda*B/v);
c1=(v*c0)/(lambda*x1);

n=1;
d=c1*(n^a)*(x1^(-n))
while d>=e2
    n=n+1
    d=c1*(n^a)*(x1^(-n))
end
N=n;
end