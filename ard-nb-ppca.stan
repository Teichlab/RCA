data {
  int <lower=0> N; // number of data points in dataset
  int <lower=0> D; // dimension
  int <lower=0> M; // maximum dimension of latent space to consider
  int x[N, D]; // data
}
parameters {
  // latent variable
  matrix[N, M] z;
  // weights parameters
  matrix[M, D] w;
  // mean parameter
  row_vector[D] mu;

  real<lower=0> sigma;
  vector<lower=0>[M] alpha;

  vector[N] det_rate;

  real<lower=0> log_phi;

}
transformed parameters {
  matrix[N, D] eta;

  eta = z * w;
}
model {
  // priors
  to_vector(z) ~ normal(0, 1);
  for (d in 1:D){
    col(w, d) ~ normal(0, sigma * alpha);
  }
  sigma ~ lognormal(0, 1);
  alpha ~ inv_gamma(1, 1);

  log_phi ~ normal(0, 1);

  // likelihood
  for (n in 1:N){
    x[n] ~ neg_binomial_2(exp(eta[n] + mu + det_rate[n] + log(sum(x[n]) + 1.)), exp(log_phi));
  }
}
