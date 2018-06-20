function update_u(X, tau, v, v2, precision_type)
    if precision_type == "rowwise_constant" || precision_type == "constant"
        s2 = 1./ (tau * sum(v2));
    else
        s2 = 1./ (tau * v2);
    end
    x = ((X .* tau) * v) .* s2;
    u = x ./ (s2 + 1);
    u2 = u.^2 .+ (s2 ./ (s2 + 1));
    return u, u2;
end

function update_v(X, tau, u, u2, precision_type, nv)
    if precision_type == "columnwise_constant" || precision_type == "constant"
        s2 = 1./ (tau' * sum(u2));
    else
        s2 = 1./ (tau' * u2);
    end
    x = ((X .* tau)' * u) .* s2;
    temp = ash(x,s2, nv = nv);
    return temp[1], temp[2]
end

function update_tau(R2, precision_type)
    if precision_type == "rowwise_constant"
        tau = 1./mean(R2,2);
    elseif precision_type == "columnwise_constant"
        tau = 1./mean(R2,1);
    elseif precision_type == "constant"
        tau = 1/mean(R2);
    elseif precision_type == "elementwise"
        tau = 1./R2;
    end    
end

function update_R2(X, X2, u, u2, v, v2)
    return u2 * v2' + X2 - 2 * (u * v') .* X
end