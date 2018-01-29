// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
	"../rand"
	"math"
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
	matrix.Require(features.Rows() != labels.Rows(),
		"CountMisclassifications: Mismatching number of rows\n")

	mis := 0
	for i := 0; i < features.Rows(); i++ {
		pred := learner.Predict(features.Row(i))
		lab := labels.Row(i)
		for j := 0; j < len(lab); j++ {
			if pred[j] != lab[j] {
				mis++
			}
		}
	}
	return mis
}

// SSE computes the sum square error.
func SSE(learner SupervisedLearner, features, labels *matrix.Matrix) float64 {
	matrix.Require(features.Rows() == labels.Rows(),
		"SSE: Mismatching number of rows\n")

	sse := 0.0
	for i := 0; i < features.Rows(); i++ {
		pred := learner.Predict(features.Row(i))
		lab := labels.Row(i)
		for j := 0; j < len(lab); j++ {
			diff := pred[j] - lab[j]
			sse += diff * diff
		}
	}
	return sse
}

// perform m-repititions n-fold cross-validation
func MRepNFoldCrossValidation(learner SupervisedLearner,
	features, labels *matrix.Matrix, m, n int) matrix.Vector {

	matrix.Require(features.Rows() == labels.Rows(),
		"MRepNFoldCrossValidation: features and labels must have the same number of rows\n")

	// partition
	start := []int{0, 0}
	end := []int{0, 0}
	rows := labels.Rows()

	// foldSize contains the size of each foldSize
	foldSize := make([]int, n)
	foldSize[0] = rows / n
	for i := 1; i < n; i++ {
		foldSize[i] = foldSize[0]
	}
	for i := 0; i < rows%n; i++ {
		foldSize[i]++
	}

	r := rand.NewRand(1982)
	var trainDataX, testDataX, trainDataY, testDataY matrix.Matrix
	sse := matrix.NewVector(m, nil)
	for i := 0; i < m; i++ {
		// shuffling data
		for j := rows; j > 1; j-- {
			l := int(r.Next(uint64(j)))
			features.SwapRows(j-1, l)
			labels.SwapRows(j-1, l)
		}

		// training

		startRemoveIndex := 0
		end[1] = rows
		sse[i] = 0.0
		for j := 0; j < n; j++ {
			// copy data into training and testing data
			start = start[:1]
			end = end[:1]
			start[0] = startRemoveIndex
			startRemoveIndex += foldSize[j]
			end[0] = startRemoveIndex
			testDataX.WrapRows(features, start, end)
			testDataY.WrapRows(labels, start, end)

			start = start[:2]
			end = end[:2]
			start[1] = end[0]
			end[0] = start[0]
			start[0] = 0
			trainDataX.WrapRows(features, start, end)
			trainDataY.WrapRows(labels, start, end)

			// train
			learner.Train(&trainDataX, &trainDataY)

			// compute SSE
			sse[i] += SSE(learner, &testDataX, &testDataY)
		}
		// average sse is
		sse[i] /= float64(rows)
		sse[i] = math.Sqrt(sse[i])
	}
	return sse
}
