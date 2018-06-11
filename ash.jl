using LowRankApprox

function mixSQP(L)
  n = size(L,1); k = size(L,2);
  F = psvdfact(L, rtol=1e-8);
  S = Diagonal(F[:S]);
  iter = 100;
  # initialize
  x = zeros(k); x[1] = 1/2; x[k] = 1/2;

  # QP subproblem start
  for i = 1:iter
    # gradient and Hessian computation -- Rank reduction method
    D = 1./(F[:U]*(S*(F[:Vt]*x)) + 1e-8);
    g = -F[:Vt]'*(S * (F[:U]'*D))/n;
    H = (F[:V]*S*(F[:U]'*Diagonal(D.^2)*F[:U])* S*F[:Vt])/n + 1e-8 * eye(k);
    # initialize
    ind = find(x .> 1e-3);
    y = sparse(zeros(k)); y[ind] = 1/length(ind);

    # Active set method start
    for j = 1:100
      # define smaller problem
      s = length(ind);
      H_s = H[ind,ind];
      d = H * y + 2 * g + 1;
      d_s = d[ind];

      # solve smaller problem
      p = sparse(zeros(k));
      p_s = -H_s\d_s; p[ind] = p_s;

      # convergence check
      if norm(p_s) < 1e-8
        # compute the Lagrange multiplier
        lambda = d - minimum(d_s);
        # convergence test
        if minimum(lambda) >= 0;
          break;
        else
          ind_min = findmin(lambda)[2];
          ind = sort([ind;ind_min]);
        end

      # do update otherwise
      else
        # retain feasibility
        alpha = 1;
        alpha_temp = -y[ind]./p_s;
        ind_block = find(p_s .< 0);
        alpha_temp = alpha_temp[ind_block];
        if ~isempty(ind_block)
          temp = findmin(alpha_temp);
          if temp[1] < 1
            ind_block = ind[ind_block[temp[2]]]; # blocking constraint
            alpha = temp[1];
            # update working set -- if there is a blocking constraint
            deleteat!(ind, find(ind - ind_block .== 0));
          end
        end
        # update
        y = y + alpha * p;
      end
    end
    x = y;

    # convergence check
    if minimum(g+1) >= 0
      break;
    end
  end
  x[x .< 1e-3] = 0
  return full(x/sum(x))
end

function ash(x,s2; samp = 2, mult = 1.2)
    x2 = x.^2;
    
    if samp == 1
        s_min = sqrt(minimum(s2))/10;
        if all(x2 .<= s2) s_max = 10 * s_min; # to deal with the occassional odd case
        else s_max = 2 * sqrt(maximum(x2-s2)); # this computes a rough largest value you'd want to use
        end

        # choose grid
        if mult == 0
            grid = [0;s_max];
        else 
            n = ceil(Int, log2(s_max/s_min)/log2(mult));
            grid = [0;mult.^((-n):0) * s_max];
        end
    else
        s_max = 2 * sqrt(maximum(x2-s2));
        ss = ceil(Int,log10(s_max))
        grid = logspace(0,ss,50*ss)-1;
        grid = grid[grid .< s_max];
    end
    
    # matrix likelihood
    s_matrix = sqrt.((s2) .+ (grid.^2)') # n by m matrix of standard deviations of convolutions
    log_lik = -(x./s_matrix).^2/2 - log.(s_matrix) - log(2*pi)/2;
    log_lik = log_lik - repmat(maximum(log_lik,2),1,size(log_lik,2));
    log_lik = exp.(log_lik);
    
    # fit the model
    p = mixSQP(log_lik);
    
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
