# Technical Notes

## Aims

1. Understand an existing network in terms of the roles of reactions.
2. Identify modules of reaction networks, networks that produce the same behavior.
3. Generate a reaction network that produces a desired timecourse.

## Network Discovery

1. SINDy has problems with scaling and interpretation because of its generality. For reaction networks, it likely suffices to consider polynomials of order 0, 1, 2. So, we could do a linear regression on terms $1, S_1, \cdots, S_N, S_1 S_1, S_1, S_2, \cdots, S_N S_N$. We just need to use an orthogonal form of the quadratic to avoid co-linearities. Also, we need to have enough data. Probably want to LASSO to provide sparsity. See Appendex A1.

# Appendices

## A1: Orthogonal Polynomials for Regression

To make linear and quadratic terms orthogonal in regression, use **orthogonal polynomials** — specifically, transform your predictors using the **Gram-Schmidt orthogonalization** applied to the polynomial basis.

---

### The Problem

For a predictor $x$, the raw basis $\{1, x, x^2\}$ is highly collinear. The orthogonal polynomial basis $\{P_0(x), P_1(x), P_2(x)\}$ satisfies:

$$\sum_{i=1}^n P_j(x_i) \cdot P_k(x_i) = 0 \quad \text{for } j \neq k$$

---

### The Transformation

Using Gram-Schmidt, construct:

$$P_0(x) = 1$$

$$P_1(x) = x - \bar{x}$$

$$P_2(x) = (x - \bar{x})^2 - \frac{\sum(x_i - \bar{x})^4}{\sum(x_i - \bar{x})^2} \cdot 1 - \frac{[\sum(x_i-\bar{x})^3]^2}{\sum(x_i-\bar{x})^2 \cdot n}$$

The simplified key results:

- **Linear term:** $\tilde{x} = x - \bar{x}$ (centering is sufficient for orthogonality between $P_0$ and $P_1$)
- **Quadratic term:** $\tilde{x}^2 = (x - \bar{x})^2 - \widehat{\text{proj}}_{P_1}[(x-\bar{x})^2]$, which removes the component of $(x-\bar{x})^2$ correlated with $P_1$

For **symmetric/uniform** $x$ distributions, $(x - \bar{x})$ and $(x - \bar{x})^2$ are already orthogonal, so **centering alone suffices**.

---

### Practical Steps

1. **Center:** $z_i = x_i - \bar{x}$
2. **Compute the correction:** $\alpha = \frac{\sum z_i^3}{\sum z_i^2}$
3. **Orthogonal quadratic:** $q_i = z_i^2 - \alpha z_i - \frac{\sum z_i^2}{n}$

Then regress on $\{1, z_i, q_i\}$.

---

### In Practice

Most statistical software handles this automatically:

| Software | Command |
|---|---|
| **R** | `poly(x, degree=2)` — returns orthogonal polynomials by default |
| **Python (numpy)** | `numpy.polynomial.polynomial.polyvander()` + QR decomposition |
| **Python (sklearn)** | `PolynomialFeatures` + center/scale manually, or use `numpy.polynomial` |
| **Stata** | `orthpoly x, deg(2) generate(p*)` |

---

### Benefits

- **Numerical stability:** eliminates multicollinearity between $x$ and $x^2$
- **Interpretability:** coefficients of $P_1$ and $P_2$ are independent — you can test the significance of the quadratic term without the linear term's coefficient changing
- **Same fitted values:** the model fit is identical to the raw polynomial regression; only the parameterization changes

The key insight is that **centering solves most of the problem** — the residual orthogonalization of the quadratic term matters most when your $x$ distribution is noticeably asymmetric.
