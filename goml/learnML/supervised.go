// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
	"../rand"
)

type SupervisedLearner interface {
	// Returns the name of this learner
	Name() string

	// Train this learner
	Train(features, labels *matrix.Matrix)

	// Partially train using a single pattern
	TrainIncremental(feat, lab matrix.Vector)

	// Make a prediction
	Predict(in matrix.Vector) matrix.Vector

	// This default implementation just copies the data, without
	// changing it in any way.
	FilterData(featIn, labIn, featOut, labOut *matrix.Matrix)
}

// CountMisclassifications measures the misclassifications with the
// provided test data.
func CountMisclassifications(learner SupervisedLearner, features, labels *matrix.Matrix) int {
	Require(features.Rows() != labels.Rows(),
		"CountMisclassifications: Mismatching number of rows\n")

	mis := 0
	for i := 0; i < features.Rows(); i++ {
		pred := learner.Predict(features.Row(i))
		lab := labels.Row(i)
		for j = 0; j < len(lab); j++ {
			if pred[j] != lab[j] {
				mis++
			}
		}
	}
	return mis
}

// CountMisclassifications measures the misclassifications with the
// provided test data.
func SSE(learner SupervisedLearner, features, labels *matrix.Matrix) float64 {
	Require(features.Rows() != labels.Rows(),
		"SSE: Mismatching number of rows\n")

	sse := float64(0)
	for i := 0; i < features.Rows(); i++ {
		pred := learner.Predict(features.Row(i))
		lab := labels.Row(i)
		for j = 0; j < len(lab); j++ {
			diff := pred[j] - lab[j]
			sse += diff * diff
		}
	}
	return sse
}
