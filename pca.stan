data {
    int<lower = 1> N;
    int<lower = 1> G;

    vector[G] Y[N];
}
parameters {
    vector[2] X[N];
    vector[G] mu;
    matrix[G, 2] W;

    real<lower = 0> s2_model;
}
model {
    for (n in 1:N){
        Y[n] ~ normal(W * X[n] + mu, s2_model);
    }

}