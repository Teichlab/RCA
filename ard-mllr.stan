
data {
  int<lower=0> N; // number of data points i n dataset
  int<lower=0> D; // dimension

  int<lower=0> P; // number of known covariates
  vector[D] x[N]; // data
  matrix[P, N] y; // Knwon covariates

}
parameters {
  // variance parameter
  real<lower=0> sigma;

  // Partial regression weights
  matrix[D, P] w_y;
  vector<lower=0>[P] beta;
}
model {
  // priors
  for (d in 1:D){
    w_y[d] ~ normal(0, sigma * beta);
  }
  sigma ~ lognormal(0, 1);
  beta ~ inv_gamma(1, 1);

  // likelihood
  for (n in 1:N){
    x[n] ~ normal (w_y * col(y, n), sigma);
  }
}
