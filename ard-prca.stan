
data {
  int<lower=0> N; // number of data points i n dataset
  int<lower=0> D; // dimension
  int<lower=0> M; // maximum dimension o f latent space t o consider
  int<lower=0> P; // number of known covariates

  vector[D] x[N]; // data
  matrix[P, N] y; // Knwon covariates

}
parameters {
  // latent variable
  matrix[M, N] z;
  // weights parameters
  matrix[D, M] w;
  // variance parameter
  real<lower=0> sigma;
  // hyper-parameters on weights
  vector<lower=0>[M] alpha;

  vector[D] mu;

  // Partial regression weights
  matrix[D, P] w_y;
  vector<lower=0>[P] beta;
}
model {
  // priors
  to_vector(z) ~ normal(0,1);
  for (d in 1:D){
    w[d] ~ normal(0, sigma * alpha);
  }
  for (d in 1:D){
    w_y[d] ~ normal(0, sigma * beta);
  }
  sigma ~ lognormal(0, 1);
  alpha ~ inv_gamma(1, 1);
  beta ~ inv_gamma(1, 1);

  // likelihood
  for (n in 1:N){
    x[n] ~ normal (w * col(z, n) + w_y * col(y, n) + mu, sigma);
  }
}
