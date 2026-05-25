# Network Discovery

This document describes how to use RSINDy to discover a reaction network from data. The initial implementation assumes the following:
* There is no noise.
* Data are provided for all chemical species.
* Rate laws for reactions are polynomials consisting of a single term with degrees of 0, 1, or 2.

## Workflow

1. Use ``NetworkRateDiscovery`` in @network_rate_discovery.py to discovery a system of differential equations for species concentrations. The equations are summarized in the dataframe returned by the ``summary`` method.
Define ``dE_i`` to be the equation for the derivative of species ``i``.
2. The foregoing fits the  *derivative* of species concentrations. We want to fit to the species actual concentrations. To do this, we add consider adding a constant to the fit of derivatives. That is, for species $i$, we find $k_i$ to minimize $J(k_i) = \sum_t ( x_i(t) - \hat{x}_i(t) - k_i)^2$, where $x_i(t)$ is the true concentration of species $i$ at time $t$, and $\hat{x}_i (t)$ is the predicted value obtained by integrating the fit in (1). Since $J(k_i)$ is convex, it suffices to find $k_i$ such that $\frac{\partial J}{\partial k_i} = 0$. That is, $k_i = \frac{1}{M} \sum_t ( x_i(t) - \hat{x}_i(t) - k_i)$, where $M$ is the number of time steps. This is implemented as follows: 
   1. For each species index ``i``
      1. Calculate $\hat{x}_i (t)$ for $t \in (t_1, \cdots, t_M)$ by integrating ``dE_i`` using $x_j(t)$ instead of $\hat{x}_j(t)$ for $j \neq i$.
      2. $k_i = \frac{1}{M} \sum_t ( x_i(t) - \hat{x}_i(t) - k_i)$.
      3. Update ``dE_i`` to include $k_i$
3. Construct the network.
   1. For each polynomial term $m$
      1. Let $V$ be the set of distinct absolute values 
      2. For v in $V$
         1. Let $R$ be the set of columns with $-v$ and $P$ the set of columns with $v$.
         2. Create a reaction with the reactions in $R$ and the products in $P$ and the rate law ``v*P_m``.
   2. Add the boundaries
      1. For each species $i$
         1. If $k_i > 0$, add a reaction that synthesizes species $i$ at the rate $k_i$.
         2. If $k_i <> 0$, add a reaction that degrades species $i$ at the rate $-k_i$. 