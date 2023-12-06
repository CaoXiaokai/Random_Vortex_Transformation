function [ degree ] = Random_Function( vor_r, idx)

syms x

weight = 2*rand(1,6)-2;

f = {};
f{1} = weight(1).*sin(weight(2).*x+weight(3)) + weight(4).*cos(weight(5).*x+weight(6));
f{2} = weight(1).* x + weight(2).*x^3 + weight(3).*exp(2*x) + weight(4).*cos(weight(5).*x+weight(6));
f{3} = weight(1).* sqrt(x) + weight(2).* x^2 + weight(3).* exp(1*x) + weight(4).*sin(weight(5).*x+weight(6)); 
f{4} = weight(1).* x^5 + weight(2).* log(x+1) + weight(3).* exp(x) + weight(4).*cos(weight(5).*x+weight(6)); 
f{5} = weight(1).* 2^x + weight(2).* (2*x) + weight(3).* log10(x+1) + weight(4).*sin(weight(5).*x+weight(6)); 
for i=0:vor_r
    degree(i+1) = double(subs(f{(rem(idx,5)+1)}, x, i/vor_r));
end

