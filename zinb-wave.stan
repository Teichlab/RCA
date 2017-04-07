
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

}
model {
  row_vector[N] mu;
  row_vector[N] pi_;
  real theta[N];

  real zeta[G];

  real delta_0;

  // likelihood
  for (n in 1:N){
    mu = exp(beta_mu[gene[n]] * x[n] + gamma_mu[gene[N]]);
    pi_ = inv_logit(beta_pi[gene[n]] * x[n] + gamma_pi[gene[N]]);
    theta[n] = exp(zeta[gene[n]]);
    if (y[n] == 0) {
      delta_0 = 1.;
    }
    else {
      delta_0 = 0.;
    }
    target += pi_[n] * delta_0 + (1. - pi_[n]) * neg_binomial_2_lpmf(y[n] | mu[n], theta[n]);
  }
}
