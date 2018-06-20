# Try to select a reasonable set of sigma values that should be used
# for the adaptive shrinkage model based on the values of x (the noisy
# observations) and s (the standard error in the observations). The
# return value is a vector of sigma values.
function autoselectmixgrid(x::Array{Float64,1};
                         nv::Int = 0,
                         gridtype::String = "mult")

  # Get the number of samples.
  n = length(x);

  # Check input "nv".
  if !(nv == 0 || nv > 1)
    throw(ArgumentError("Input \"nv\" should be 0, or greater than 1"))
  end
    
  # This is, roughly, the largest value you'd want to use.
  sigmamax = 2 * sqrt(maximum(x.^2 - 1)); 
  
  # If maximum(x.^2 - 1) < 1 then probably it is natural to say each x is null
  
  if nv > 0
      # Choose the grid of sigmas.
      if gridtype == "mult" # multiplicative grid
        return logspace(0,log10(sigmamax),nv) - 1;
      elseif gridtype == "add" # additive grid
        return linspace(0,sigmamax,nv);
      else
        throw(ArgumentError("Input \"gridtype\" should be \"mult\" or \"add\"."));
      end
  else
      return 0:ceil(sigmamax)
  end
end