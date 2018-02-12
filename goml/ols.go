package main

import (
	"./learnML"
	"./matrix"
	"./rand"
	"fmt"
)

func testOLS() {
	fmt.Printf("Testing OLS...\n")
	r := rand.NewRand(192)
	weights := matrix.NewMatrix(5, 1, nil)
	weights.Random()
	features := matrix.NewMatrix(10, 4, nil)
	features.Random(111)

	labels := matrix.NewMatrix(10, 1, nil)
	dlabel := matrix.NewMatrix(10, 1, nil)

	l := learnML.NewLayer(learnML.LayerLinear,
		learnML.Dimension{labels.Cols()},
		learnML.Dimension{features.Cols()})
	var noiseDeviation float64 = 0.1
	for i := 0; i < labels.Rows(); i++ {
		row := features.Row(i)
		copy(labels.Row(i), *(l.Activate(weights.ToVector(), &row)))
		dlabel.Row(i)[0] = labels.Row(i)[0] + noiseDeviation*r.Normal()
	}

	newWeights := matrix.OLS(features, dlabel)
	diffW := newWeights.Sub(weights.ToVector()).Norm(0)
	tol := 0.05
	fmt.Println("Result is:")
	fmt.Printf("max|newWeights - weights| = %e where tolerance = %e\n\n",
		diffW, tol)
	if diffW > tol {
		panic("These weights are too different!")
	}

	n := learnML.NewNeuralNet([]int{labels.Cols()},
		learnML.LayerLinear, features.Cols(), labels.Cols())
	n.InitWeight(nil)
	gradient := n.CreateGradient()
	rate := 0.05
	var N int = 100
	for i := 0; i < N; i++ {
		learnML.ResetGradient(gradient)
		for j := 0; j < features.Rows(); j++ {
			x := features.Row(j)
			y := dlabel.Row(j)
			n.Activate(matrix.Vector{}, &x)
			n.BackProp(y, nil)
			n.UpdateGradient(&x, gradient)
		}
		n.RefineWeight(gradient, rate)
	}
	_, newWeights2 := n.Weight()
	diffW = newWeights2.Sub(newWeights).Norm(0)
	fmt.Println("Difference between OLS and Gradient Descent:")
	fmt.Printf("Learning rate = %e and number of epochs is %d\n",
		rate, N)
	fmt.Printf("max|newWeights2 - newWeights| = %e\n\n", diffW)
	if diffW > tol {
		panic("These weights are too different!")
	}
}

func testMRepNFoldCV(m, n int) {
	fmt.Printf("Testing %d-repetitions %d-fold cross-validation ...\n",
		m, n)

	var features, labels matrix.Matrix
	var sse matrix.Vector

	features.LoadARFF("housing_features.arff")
	labels.LoadARFF("housing_labels.arff")
	// learnML.LayerLinear{} is an instance of type learmML.LayerLinear
	N := learnML.NewNeuralNet([]int{1}, learnML.LayerLinear, features.Cols(), labels.Cols())

	sse = learnML.MRepNFoldCrossValidation(N, &features, &labels, m, n)
	fmt.Printf("RMSE are %v\n", sse)
}

func main() {
	testOLS()
	//testMRepNFoldCV(5, 10)
}
