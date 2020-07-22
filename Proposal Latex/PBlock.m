function Pblck=PBlock(c,K,lambda,mu,xi,v)

N=findN(lambda,mu,xi,v);

Q1=matQ1(c,K,lambda,mu,xi,v,0);
Q0=matQ0(c,K,lambda);
A=Q0+Q1;
O=zeros((c+1)*(K+1),1); 
e=ones((K+1)*(c+1),1);
x0=linprog(e,[],[],A',O,O,e);

sum=x0'*e;

xbaru=x0/sum;
s=xbaru'*e;

PBlock=0
for i=1:c+1
    PBlock=PBlock+xbaru(i+(c+1)*K)
end
end