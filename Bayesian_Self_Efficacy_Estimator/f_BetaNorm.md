


Given parameters $\alpha$ and $\beta$, and a variable $x$, the function `f_BetaNorm` computes the normalized Beta distribution as:

$$
f_{\text{BetaNorm}}(\alpha, \beta, x) = \frac{f(x; \alpha, \beta)}{f_{\text{max}}}
$$

where

$$
f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

is the probability density function of the Beta distribution, $B(\alpha, \beta)$ is the Beta function, and

$$
f_{\text{max}} = f\left(\frac{\alpha-1}{\alpha+\beta-2}; \alpha, \beta\right)
$$

is the maximum value of the Beta distribution.


The Beta function $B(\alpha, \beta)$ is defined in terms of the Gamma function $\Gamma$, which is a generalization of the factorial function to complex numbers. The Beta function can be expressed as:

$$
B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}
$$

So, the function `f_BetaNorm` can be written in full as:

$$
f_{\text{BetaNorm}}(\alpha, \beta, x) = \frac{\frac{x^{\alpha-1}(1-x)^{\beta-1}}{\frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}}}{\frac{\left(\frac{\alpha-1}{\alpha+\beta-2}\right)^{\alpha-1}\left(1-\frac{\alpha-1}{\alpha+\beta-2}\right)^{\beta-1}}{\frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}}}
$$

This simplifies to:

$$
f_{\text{BetaNorm}}(\alpha, \beta, x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{\left(\frac{\alpha-1}{\alpha+\beta-2}\right)^{\alpha-1}\left(1-\frac{\alpha-1}{\alpha+\beta-2}\right)^{\beta-1}}
$$
