package main

import (
	"./learnML"
	"./matrix"
	"./rand"
	"fmt"
	"time"
)

func testOLS() {
	fmt.Printf("Testing OLS...\n")
	r := rand.NewRand(192)
	weights := matrix.NewMatrix(5, 1, nil)
	weights.Random()
	features := matrix.NewMatrix(10, 4, nil)
	features.Random(111234)

	labels := matrix.NewMatrix(10, 1, nil)
	dlabel := matrix.NewMatrix(10, 1, nil)

	l := learnML.NewLinearLayer(labels.Cols(), features.Cols(), weights.ToVector())
	var noiseDeviation float64 = 0.1
	for i := 0; i < labels.Rows(); i++ {
		row := features.Row(i)
		copy(labels.Row(i), *(l.Activate(&row)))
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
	for i := 0; i < 100; i++ {
		n.Train(features, dlabel)
	}
	_, newWeights2 := n.Weight()
	diffW = newWeights2.Sub(newWeights).Norm(0)
	fmt.Println("Difference between OLS and Gradient Descent:")
	fmt.Println("learning rate .03, after 100 epochs")
	//fmt.Printf("max|newWeights2 - newWeights| = %e\n\n",
	fmt.Printf("max|weight(GD) - weight(OLS)| = %e\n\n",
		diffW/newWeights.Norm(2))
	fmt.Printf("weight(GD) = %v and weight(OLS) = %v\n", newWeights2, newWeights)
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
	N := learnML.NewNeuralNet([]int{1}, learnML.LayerLinear,
		features.Cols(), labels.Cols())

	sse = learnML.MRepNFoldCrossValidation(N, &features, &labels, m, n)
	fmt.Printf("after N is %v\n", N)
	fmt.Printf("RMSE are %v and row = %d\n", sse, features.Rows())
}

func mnist(numPeriod int) {
	var features, labels, testFeatures, testLabels matrix.Matrix

	features.LoadARFF("/home/hoangngan/courses/MIS/machineLearning/mnist/train_feat.arff")
	labels.LoadARFF("/home/hoangngan/courses/MIS/machineLearning/mnist/train_lab.arff")
	testFeatures.LoadARFF("/home/hoangngan/courses/MIS/machineLearning/mnist/test_feat.arff")
	testLabels.LoadARFF("/home/hoangngan/courses/MIS/machineLearning/mnist/test_lab.arff")
	features.Scale(1.0 / 256.0)
	testFeatures.Scale(1.0 / 256.0)

	// convert labels to hot-spot representation of a number which is
	// a list of 10 binary value. If lables[i] = 8 then the 7-th
	// position of the string will be set (=1). Otherwise, they are all
	// 0's.
	mlabels := matrix.NewMatrix(labels.Rows(), 10, nil)
	for i := 0; i < labels.Rows(); i++ {
		mlabels.SetElem(i, int(labels.GetElem(i, 0)), 1.0)
	}

	n := learnML.NewNeuralNet([]int{80}, learnML.LayerTanh,
		features.Cols(), mlabels.Cols())
	t := learnML.NewLayer(learnML.LayerLeakyRectifier, []int{30})
	n.AddLayer(t)
	t = learnML.NewLayer(learnML.LayerTanh, []int{10})
	n.AddLayer(t)
	//n = learnML.NewNeuralNet([]int{98, 10}, learnML.LayerTanh,
	//features.Cols(), mlabels.Cols())
	n.InitWeight(nil)
	var mis int
	start := time.Now()
	innerStart := time.Now()
	for i := 0; i < numPeriod; i++ {
		fmt.Printf("Training %2d:... ", i)
		innerStart = time.Now()
		n.Train(&features, mlabels)
		fmt.Printf("%5.2fs\tCounting Misclassifications:... ",
			time.Since(innerStart).Seconds())
		mis = 0
		for i := 0; i < testFeatures.Rows(); i++ {
			pred := n.Predict(testFeatures.Row(i))
			val := 0
			max := 0.0
			for j := 0; j < len(pred); j++ {
				if pred[j] > max {
					max = pred[j]
					val = j
				} else {
					if pred[j] < -max {
						max = -pred[j]
						val = j
					}
				}
			}
			if val != int(testLabels.GetElem(i, 0)) {
				mis++
			}
		}
		fmt.Printf("%d\t%7.2fs\n", mis, time.Since(start).Seconds())
	}
}

func main() {
	testOLS()
	//testMRepNFoldCV(1, 10)
	fmt.Println("\n\nMNIST training with MLP - LayerTanh\n")
	mnist(100)
}
