# Two cases:
# (1) normal prior/normal likelihood
# (2) truncated normal prior/normal likelihood


# Compute the n x k conditional likelihood matrix, where n is the
# number of samples and k is the number of mixture components,

# for the case (1)
# when the likelihood is univariate normal and prior is a mixture
# of univariate normals.
# entry (i,j) of the conditional likelihood matrix is equal to
# N(0,s[i]^2 + sd[j]^2), the normal density with zero mean and
# variance s[i]^2 + sd[j]^2.

# for the case(2)
#


# If normalize = true, each row of the likelihood matrix is divided by
# the largest value in the row. After normalization, the largest value
# in each row is 1.
function matrixlik( x::Array{Float64,1};
                     grid::Array{Float64,1} = autoselectmixgrid(x),
                     mixtype::String = "trucnorm",
                     normalizerows::Bool = true)

  # Get the number of samples (n) and the number of prior mixture
  # components (k).
  n = length(x);
  m = length(grid);

  # Check input "grid".
  if any(grid .< 0)
    throw(ArgumentError("All elements of \"grid\" should be non-negative"))
  end
  
  # Compute the n x m matrix of standard deviations.
  S = sqrt.(1 + grid.^2);
  
  # Compute the log-densities, and normalize the rows, if requested.
  if mixtype == "norm"
      L = -(x./S).^2/2 - log.(S) - log(2*pi)/2;
        
  elseif mixtype == "truncnorm"
      X = x .* (grid ./ S)'
      L = -log.(1+erf.(X/sqrt(2))) .* (((x.^2)./S')/2 .+ log.(S)')
  end
    
  if normalizerows

    # This is the same as
    #
    #   L = L - repmat(maximum(L,2),1,k);
    #
    # but uses memory more efficiently to complete the operation.
    L = broadcast(-,L,maximum(L,2));
  end
  return exp.(L)
end