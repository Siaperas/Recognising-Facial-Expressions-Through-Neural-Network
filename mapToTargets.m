function [T] = mapToTargets(y)
%Function that maps 1xn vector y from range 1-6 to 6xn matrix with each
%row consisting of a 1x6 binary vector with element y(i) = 1 and all others
%0
    n = size(y,1);
    T = zeros(n,6);
    for i=1:n
        T(i,y(i)) = 1;
    end
end