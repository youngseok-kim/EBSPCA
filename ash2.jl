function ash2(x,s2; nv_max = 50)

    n = length(x); flag = 0;
    x2_div_s2 = x.^2 ./ s2;
    s2_max = 4 * maximum(x2_div_s2 - 1);
    
    if s2_max <= 0
        flag == 1; # flag 1 means ash will output 0 solution
        return 0, 0, 0, 0, 0, 0, 0, flag
    end
    
    s2_max_round = ceil(Int,log10(s2_max))
    grid = logspace(0, s2_max_round, nv_max) - 1;
    #grid = grid[grid .< s2_max_round];
    
    # matrix likelihood
    s2_matrix = 1 + (grid.^2)' # n by m matrix of standard deviations of convolutions
    log_lik = -x2_div_s2./s2_matrix .- log.(s2_matrix);
    log_lik = log_lik - repmat(maximum(log_lik,2),1,size(log_lik,2));
    log_lik = exp.(log_lik);
    
    # fit the model
    p = mixSQP(log_lik)["x"];
    
    # exploit sparsity
    ind = find(p .> 0);
    
    if length(ind) == 0
        flag == 1; # flag 1 means ash outputs 0 solution
        return 0, 0, 0, 0, 0, 0, 0, flag
    end
    
    ind = unique([1;ind]);
    
    ps2 = grid[ind];
    
    # posterior calculation
    temp = sqrt.(s2) * (1 + ps2)';
    comp_post_mean = (x * ps2') ./ temp;
    comp_post_sd2 = (s2 * ps2') ./ temp;
    comp_post_mean2 = comp_post_sd2 + comp_post_mean.^2;
    comp_post_prob = log_lik[:,ind] .* p[ind]';
    comp_post_prob = comp_post_prob ./ sum(comp_post_prob,2);
    post_mean = sum(comp_post_prob .* comp_post_mean,2);
    post_mean2 = sum(comp_post_prob .* comp_post_mean2,2);
    
    # return posterior first/second moments
    return post_mean, post_mean2, log_lik, p, comp_post_prob, comp_post_mean, comp_post_sd2, flag

end
