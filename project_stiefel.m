function Q = project_stiefel( Q )
    % this is a well known projection, see eg Manton or Fan and Hoffman 1955
%     [U,S,V] = svd(Q,0);
    dimD = size(Q,2);
    [U, S, V] = mySVD(Q,dimD);
    Q = U*V';
end

    
