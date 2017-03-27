
data {
  int < lower=0> N; // number of data points i n dataset
  int < lower=0> D; // dimension
  int < lower=0> M; // maximum dimension o f latent space t o consider
  vector [D] x[N]; // data
}
parameters {
  // latent variable
  matrix[M, N] z;
  // weights parameters
  matrix[D, M] w;
  // variance parameter
  real<lower=0> sigma;
  // mean parameter
  vector[D] mu;
  // hyper - parameters on weights
  vector<lower=0 >[M] alpha;
}
model {
  // priors
  to_vector (z) ~ normal(0,1);
  for (d in 1:D){
    w[d] ~ normal(0, sigma * alpha);
  }
  sigma ~ lognormal(0, 1);
  alpha ~ inv_gamma(1, 1);

  // likelihood
  for (n in 1:N){
    x[n] ~ normal(w * col(z, n) + mu, sigma);
  }
}
