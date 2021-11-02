function [ x ] = normalize1( x )
%NORMALIZED Summary of this function goes here
%   Detailed explanation goes here
for i=1:size(x,1)
    if(norm(x(i,:))==0)    
        
    else
        x(i,:)=x(i,:)./norm(x(i,:));
    end
end

