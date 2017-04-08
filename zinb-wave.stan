
data {
  int<lower=0> N; // number of data points in dataset
  int<lower=1> P; // number of known covariates
  int<lower=1> K; // number of hidden dimensions
  int<lower=1> G; // number of observed genes
  int<lower=1> C; // number of observed cells

  vector[P] x[N]; // Covariates, including intercept.
  int y[N];      // Expression values (counts!)
  int<lower=1, upper=G> gene[N]; // Gene identifiers
  int<lower=1, upper=G> cell[N]; // Cell identifiers

}
parameters {
  // Latent variable model
  matrix[G, K] alpha_mu;
  matrix[G, K] alpha_pi;

  matrix[K, C] w;

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
  row_vector[1] mu;
  row_vector[1] pi_;
  real theta;

  // likelihood
  for (n in 1:N){
    mu = exp(beta_mu[gene[n]] * x[n] + gamma_mu[gene[N]] + alpha_mu[gene[n]] * col(w, cell[n]));
    pi_ = beta_pi[gene[n]] * x[n] + gamma_pi[gene[N]] + alpha_pi[gene[n]] * col(w, cell[n]);
    theta = exp(zeta[gene[n]]);

    if (y[n] > 0) {
      target += bernoulli_logit_lpmf(0 | pi_) + neg_binomial_2_lpmf(y[n] | mu, theta);
    }
    else {
      target += log_sum_exp(bernoulli_logit_lpmf(1 | pi_),
                            bernoulli_logit_lpmf(0 | pi_) + neg_binomial_2_lpmf(y[n] | mu, theta));
    }
  }
}
