function ash(x,s2; nv = 10, nullprior = 10)
    x2 = x.^2;
    
    if length(s2) == 1
        s2 = s2 .* ones(length(x))
    end
    

    s_max = 2 * maximum(x2-s2);
    ss = ceil(Int,log10(s_max))
    grid = logspace(0,ss,nv)-1;
    
    # matrix likelihood
    s_matrix = sqrt.(s2 .+ grid') # n by m matrix of standard deviations of convolutions
    log_lik = -(x./s_matrix).^2/2 - log.(s_matrix);
    log_lik = log_lik - repmat(maximum(log_lik,2),1,size(log_lik,2));
    log_lik = exp.(log_lik);
    
    # fit the model
    p = mixSQP(log_lik, nullprior = nullprior)["x"];
    
    # exploit sparsity
    ind = find(p .> 0);
    ind = unique([1;ind]); # we don't need this if nullprior > 0, since the null component never gonna be 0
    ps2 = grid[ind];
    
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
    return Dict([
                (:pm, post_mean), (:pm2, post_mean2), (:ll, log_lik), (:pp, p),
                (:cpp, comp_post_prob), (:cpm, comp_post_mean), (:cps2, comp_post_sd2),
                (:x, x), (:s2, s2), (:grid, grid), (:ps2, ps2)
                ])

end