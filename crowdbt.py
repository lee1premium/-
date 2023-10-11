"""Implements CrowdBT that computes scores for objects in pairwise comparisons.

References:
* Pairwise ranking aggregation in a crowdsourced setting http://goo.gl/g1q1sL.
* Summary: Ranking problem based on pairwise preferences http://goo.gl/oFv45c.
* Crowdsourced top-k algorithms: experimental evaluations http://goo.gl/q3r8xs.
"""

import math
import numpy as np
from scipy import optimize
from scipy import stats

# TODO(henrylee): initialize preference accuracies by annotator as
# the ratios of the correct answers on a handful of gold samples
# rather than the scalar value (default: 0.9) for all annotators.


def estimate_bt_ranks(graphs,
                      preference_accuracy=0.9,
                      regularization_strength=0.5):
  """Takes preferences of pairwise comparisons, and returns object indices.

  Args:
    graphs: Preferences (a.k.a pairwise comparisons) in a list of graphs
            indexed by annotators, where each graph is a set of edges, and
            an edge (j, i) states that object i is preferred over object j.
            e.g., [{(1, 0), (2, 1)}, {(3, 2)}] tells us that there are three
            pairwise comparisons by two annotators, and an annotator prefers
            object 0 over 1, and object 1 over 2, and the other annotator
            prefers object 2 over 3.
    preference_accuracy: Preference accuracy (default: 0.9), while 0.5 and 0.0
                         seem reasonable for spammers and malicious annotators.
                         This should become a vector that contains a preference
                         accuracy per annotator.
    regularization_strength: This strength is empirically in a range [0.1, 10].
                             See also: Regularization http://shortn/_hVvgbMBIn6.

  Returns:
    Object indices in the order of BT scores (from greatest to least scores).
  """
  bt_scores = estimate_bt_scores(graphs, False, preference_accuracy,
                                 regularization_strength)
  return np.add((-bt_scores).argsort().argsort(), 1)


def estimate_bt_scores(graphs,
                       standardize_bt_scores=False,
                       preference_accuracy=0.9,
                       regularization_strength=0.5):
  """Takes preferences of pairwise comparisons, and returns object scores.

  Args:
    graphs: Preferences (a.k.a pairwise comparisons) in a list of graphs
            indexed by annotators, where each graph is a set of edges, and
            an edge (j, i) states that object i is preferred over object j.
            e.g., [{(1, 0), (2, 1)}, {(3, 2)}] tells us that there are three
            pairwise comparisons by two annotators, and an annotator prefers
            object 0 over 1, and object 1 over 2, and the other annotator
            prefers object 2 over 3.
    standardize_bt_scores: a flag to standardize BT scores (default: False).
    preference_accuracy: Preference accuracy (default: 0.9), while 0.5 and 0.0
                         seem reasonable for spammers and malicious annotators.
                         This should become a vector that contains a preference
                         accuracy per annotator.
    regularization_strength: This strength is empirically in a range [0.1, 10].
  Returns:
    Object scores in Bradley-Terry model.
  """

  def bt(s_i, s_j):
    return math.exp(s_i) / (math.exp(s_i) + math.exp(s_j))

  # pylint: disable=g-bad-name
  def negative_log_likelihood(s):  # `s` contains scores for the objects.
    h = preference_accuracy  # annotator's accuracy; symbolized as h[k] (eta).
    l = regularization_strength  # within [0.1, 10]; symbolized as l (lambda).
    s_vo = 1.0  # has the score (s0) for a virtual object node (o0).
    L = sum(math.log10(h * bt(s[i], s[j]) + (1 - h) * bt(s[j], s[i]))
            for edges in graphs for j, i in edges)  # log-likelihood (L).
    R = sum(math.log10(bt(s_vo, s[i])) + math.log10(bt(s[i], s_vo))
            for i in range(len(s)))  # regularization term (R).
    return -1 * (L + l * R)  # negative, regularized log-likelihood to minimize.

  num_vertices = 1 + max(max(i, j) for edges in graphs for j, i in edges)
  initial_guess = [1.0] * num_vertices

  # TODO(henrylee): a flag for maxiter.
  # Minimizing the negative log likelihood maximizes the log likelihood.
  raw_bt_scores = np.exp(optimize.minimize(fun=negative_log_likelihood,
                                           x0=initial_guess,
                                           method='BFGS',
                                           options={'maxiter': 99}).x)
  if standardize_bt_scores:
    return stats.zscore(raw_bt_scores)
  else:
    return raw_bt_scores
