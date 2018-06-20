function ash(x,s2; nv = 50)
    x2 = x.^2;
    
    if length(s2) == 1
        s2 = s2 .* ones(length(x))
    end
    

    s_max = 2 * sqrt(maximum(x2-s2));
    ss = ceil(Int,log10(s_max))
    grid = logspace(0,ss,nv)-1;
    
    # matrix likelihood
    s_matrix = sqrt.((s2) .+ (grid.^2)') # n by m matrix of standard deviations of convolutions
    log_lik = -(x./s_matrix).^2/2 - log.(s_matrix) - log(2*pi)/2;
    log_lik = log_lik - repmat(maximum(log_lik,2),1,size(log_lik,2));
    log_lik = exp.(log_lik);
    
    # fit the model
    p = mixSQP(log_lik)["x"];
    
    # exploit sparsity
    ind = find(p .> 0);
    ps2 = grid[ind].^2;
    
    # posterior calculation
    temp = s2 .+ ps2';
    comp_post_mean = (x * ps2') ./ temp;
    comp_post_sd2 = (s2 * ps2') ./ temp;
    comp_post_mean2 = comp_post_sd2 + comp_post_mean.^2;
    comp_post_prob = log_lik[:,ind] .* p[ind]';
    comp_post_prob = comp_post_prob ./ sum(comp_post_prob,2);
    post_mean = sum(comp_post_prob .* comp_post_mean,2);
    post_mean2 = sum(comp_post_prob .* comp_post_mean2,2);
    
    # return posterior first/second moments
    return post_mean, post_mean2, log_lik, p, comp_post_prob, comp_post_mean, comp_post_sd2

end