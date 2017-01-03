data {
    int<lower = 1> N;
    int<lower = 1> G;

    vector[G] Y[N];
}
transformed data{
    vector[2] O;
    matrix[2, 2] I;
    
    O[1] = 0.;
    O[2] = 0.;

    I[1, 1] = 1.;
    I[1, 2] = 0.;
    I[2, 1] = 0.;
    I[2, 2] = 1.;
}
parameters {
    vector[2] X[N];
    vector[G] mu;
    matrix[G, 2] W;

    real<lower = 0> s2_model;
}
model {
    for (n in 1:N){
        X[n] ~ multi_normal(O, I);
    }

    for (n in 1:N){
        Y[n] ~ normal(W * X[n] + mu, s2_model);
    }
}
