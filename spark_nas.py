from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK
from hyperopt.pyll import scope
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

from hyperopt import fmin, tpe, hp,  STATUS_OK, Trials

def objective(C):
    # Create a support vector classifier model
    clf = RandomForestClassifier(max_depth=C, random_state=0)
    
    # Use the cross-validation accuracy to compare the models' performance
    accuracy = cross_val_score(clf, X, y).mean()
    
    # Hyperopt tries to minimize the objective function. A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

search_space =  scope.int(hp.uniform('C', 1, 10))

algo=tpe.suggest

"""Single Node"""

argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=16)

print(argmin)
print("```````````````````````````````````````````````````````'")

print('MULTI')

"""Multi-node
"""

from hyperopt import SparkTrials
import mlflow
spark_trials = SparkTrials()
 
with mlflow.start_run():
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)
