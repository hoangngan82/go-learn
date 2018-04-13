package main

import (
	"./learnML"
	"./matrix"
	//"./rand"
	"fmt"
	"math"
	//"time"
)

func a5(l1, l2 learnML.Dims, epoch, learningRate, tol float64) {
	features_vec := matrix.NewVector(357, nil)
	test_features_vec := matrix.NewVector(101, features_vec[256:])
	for i := 0; i < len(features_vec); i++ {
		features_vec[i] = float64(i) / 256.0
	}
	features := matrix.NewMatrix(256, 1,
		features_vec[:256])

	var labels matrix.Matrix
	labels.LoadARFF("./labor_stats.arff")
	labels_vec := labels.ToVector()[:357]
	train_labels := matrix.NewMatrix(256, 1, labels_vec[:256])
	test_labels := matrix.NewMatrix(101, 1, labels_vec[256:])

	n := learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, learnML.Dims{1, 101})
	n.AddLayer(learnML.LayerSinusoidal, learnML.Dims{100, 1})
	n.AddLayer(learnML.LayerLinear, learnML.Dims{101, 1}, l1, l2)

	n.InitWeight(nil)
	w := n.CopyWeight()
	for i := 0; i < 100; i++ {
		w[0][i] = float64((i%50)+1) * 2.0 * math.Pi
	}
	w[0][100] = 0.01
	w[0][201] = 0.0
	for i := 0; i < 50; i++ {
		w[0][i+101] = math.Pi
		w[0][i+151] = math.Pi / 2.0
	}

	params := make(map[string]float64)
	params["learningRate"] = learningRate
	params["epochPerPeriod"] = epoch
	params["maxrun"] = 1e4
	params["tolerance"] = tol
	test_features := matrix.NewMatrix(len(test_features_vec),
		1, test_features_vec)
	n.Train(features, train_labels, test_features, test_labels, params)

	results := matrix.NewMatrix(len(features_vec), 1, nil)
	features = matrix.NewMatrix(len(features_vec), 1, features_vec)
	for i := 0; i < results.Rows(); i++ {
		copy(results.Row(i), n.Predict(features.Row(i)))
	}
	if l1[0] != 0 {
		l1[0] = 1
	}
	if l2[0] != 0 {
		l2[0] = 1
	}
	s := fmt.Sprintf("/tmp/a5_%d_%d", l1[0], l2[0])
	results.SaveARFF(s, false)
}

func main() {
	// without regularization
	learningRate := .001
	tol := 1e-2
	fmt.Printf("without regularization and learning rate = %e\n", learningRate)
	//a5([]int{0, 1}, []int{0, 1}, 10.0, learningRate, tol)
	// with L1 regularization
	l10 := 1
	l11 := 8
	learningRate = .001
	fmt.Printf("L1-regularization %e and learning rate = %e\n",
		float64(l10)/float64(l11), learningRate)
	//a5([]int{l10, l11}, []int{0, 1}, 10.0, learningRate, tol)
	// with L2 regularization
	l20 := 4
	l21 := 5
	tol = 1e-3
	learningRate = .001
	fmt.Printf("L2-regularization %e and learning rate = %e\n",
		float64(l20)/float64(l21), learningRate)
	a5([]int{0, 1}, []int{l20, l21}, 10.0, learningRate, tol)
	l10 = 1
	l11 = 8
	l20 = 3
	l21 = 5
	a5([]int{l10, l11}, []int{l20, l21}, 10.0, learningRate, tol)
}
