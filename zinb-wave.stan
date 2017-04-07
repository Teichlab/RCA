
data {
  int<lower=0> N; // number of data points in dataset
  int<lower=1> P; // number of known covariates
  int<lower=1> G; // number of observed genes

  vector[P] x[N]; // Covariates, including intercept.
  int y[N];      // Expression values (counts!)
  int<lower=1, upper=G> gene[N]; // Gene identifiers

}
parameters {
  // (For now, ignore hidden latent)

  // Cell regression weights
  matrix[G, P] beta_mu;
  matrix[G, P] beta_pi;

  // Gene regression weights
  // (For now only do intercept)
  matrix[G, 1] gamma_mu;
  matrix[G, 1] gamma_pi;

  // Dispersion
  real zeta[G];

}
model {
  real mu[N];
  real pi_[N];
  real theta[N];

  // likelihood
  for (n in 1:N){
    mu[n] = exp(beta_mu[gene[n]] * x[n] + gamma_mu[gene[N]])[1];
    pi_[n] = inv_logit(beta_pi[gene[n]] * x[n] + gamma_pi[gene[N]])[1];
    theta[n] = exp(zeta[gene[n]]);
    if (y[n] != 0) {
      target += bernoulli_lpmf(0 | pi_[n]) + neg_binomial_2_lpmf(y[n] | mu[n], theta[n]);
    }
    else {
      target += log_sum_exp(bernoulli_lpmf(1 | pi_[n]),
                            bernoulli_lpmf(0 | pi_[n]) + neg_binomial_2_lpmf(y[n] | mu[n], theta[n]));
    }
  }
}
