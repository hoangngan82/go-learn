// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
  "./matrix"
  "./rand"
)

type SupervisedLearner interface {
  // Returns the name of this learner
  Name() string

  // Train this learner
  Train(features, labels *Matrix)

  // Partially train using a single pattern
  TrainIncremental(feat, lab Vector)

  // Make a prediction
  Predict(in Vector) Vector

  // Measures the misclassifications with the provided test data
  CountMisclassifications(features, labels *Matrix)

  // This default implementation just copies the data, without
  // changing it in any way.
  Filter_data(feat_in, lab_in, feat_out, lab_out *Matrix)
}
