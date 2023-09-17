# ENet plus
Elastic net that works simlilar to `sklearn.linear_model.ElasticNet` with
- the ability to bound the learned coefficients;
- finer control over what coefficients get penalized;
- the ability to add extra penalties
The package essentially runs `scipy.optimize(method='lbfgs', ...)` in the background.
