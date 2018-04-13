package main

import (
	"./learnML"
	"./matrix"
	//"./rand"
	"fmt"
	"math"
	"time"
)

func overfitting(params map[string]float64) {
	features_vec := matrix.NewVector(357, nil)
	for i := 0; i < len(features_vec); i++ {
		features_vec[i] = float64(i) / 256.0
	}

	var labels matrix.Matrix
	labels.LoadARFF("./labor_stats.arff")
	labels_vec := labels.ToVector()[:357]
	train_labels := matrix.NewMatrix(256, 1, labels_vec[:256])
	test_labels := matrix.NewMatrix(101, 1, labels_vec[256:])
	train_features := matrix.NewMatrix(256, 1, features_vec[:256])
	test_features := matrix.NewMatrix(101, 1, features_vec[256:])

	n := learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, learnML.Dims{1, 101})
	n.AddLayer(learnML.LayerSinusoidal, learnML.Dims{100, 1})
	n.AddLayer(learnML.LayerLinear, learnML.Dims{101, 1}, []int{0, 1}, []int{0, 1})

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
	n.InitWeight(w)

	maxrun := 200
	predict := matrix.NewMatrix(test_labels.Rows(), test_labels.Cols(), nil)
	train := matrix.NewMatrix(train_labels.Rows(), train_labels.Cols(), nil)

	// result matrix: errors are MSE error
	// - first column is training error
	// - second columns is validation error
	result := matrix.NewMatrix(maxrun, 2, nil)
	var err float64
	for i := 0; i < maxrun; i++ {
		n.Train(train_features, train_labels, params)

		for e := 0; e < test_features.Rows(); e++ {
			pred := n.Predict(test_features.Row(e))
			copy(predict.Row(e), pred)
		}
		for e := 0; e < train_features.Rows(); e++ {
			pred := n.Predict(train_features.Row(e))
			copy(train.Row(e), pred)
		}
		err = train.ToVector().Sub(train_labels.ToVector()).Norm(2)
		result.SetElem(i, 0, err*err/float64(256))
		err = predict.ToVector().Sub(test_labels.ToVector()).Norm(2)
		result.SetElem(i, 1, err*err/(float64(101)))
	}
	result.SaveARFF("/tmp/overfit.dat", false)
}

func batch_momentum(params map[string]float64) {
	features_vec := matrix.NewVector(357, nil)
	for i := 0; i < len(features_vec); i++ {
		features_vec[i] = float64(i) / 256.0
	}

	var labels matrix.Matrix
	labels.LoadARFF("./labor_stats.arff")
	labels_vec := labels.ToVector()[:357]
	train_labels := matrix.NewMatrix(256, 1, labels_vec[:256])
	test_labels := matrix.NewMatrix(101, 1, labels_vec[256:])
	train_features := matrix.NewMatrix(256, 1, features_vec[:256])
	test_features := matrix.NewMatrix(101, 1, features_vec[256:])

	n := learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, learnML.Dims{1, 101})
	n.AddLayer(learnML.LayerSinusoidal, learnML.Dims{100, 1})
	n.AddLayer(learnML.LayerLinear, learnML.Dims{101, 1}, []int{0, 1}, []int{0, 1})

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

	maxrun := 200
	predict := matrix.NewMatrix(test_labels.Rows(), test_labels.Cols(), nil)
	train := matrix.NewMatrix(train_labels.Rows(), train_labels.Cols(), nil)

	// result matrix: errors are RMSE error
	// - first column is training error with momentum
	// - second column is validation error with momentum
	// - third column is time
	// - forth column is training error with batch
	// - fifth column is validation error with batch
	// - sixth column is time
	result := matrix.NewMatrix(maxrun, 6, nil)
	var err float64
	start := time.Now()

	// batch
	params["batchSize"] = 16
	n.InitWeight(w)
	n.Train(train_features, train_labels, params)
	result.SetElem(0, 2, time.Since(start).Seconds())
	for e := 0; e < test_features.Rows(); e++ {
		pred := n.Predict(test_features.Row(e))
		copy(predict.Row(e), pred)
	}
	for e := 0; e < train_features.Rows(); e++ {
		pred := n.Predict(train_features.Row(e))
		copy(train.Row(e), pred)
	}
	err = train.ToVector().Sub(train_labels.ToVector()).Norm(2)
	result.SetElem(0, 0, err/(float64(16)))
	err = predict.ToVector().Sub(test_labels.ToVector()).Norm(2)
	result.SetElem(0, 1, err/math.Sqrt(float64(101)))

	for i := 1; i < maxrun; i++ {
		start = time.Now()
		n.Train(train_features, train_labels, params)
		result.SetElem(i, 2, time.Since(start).Seconds()+result.GetElem(i-1, 2))
		for e := 0; e < test_features.Rows(); e++ {
			pred := n.Predict(test_features.Row(e))
			copy(predict.Row(e), pred)
		}
		for e := 0; e < train_features.Rows(); e++ {
			pred := n.Predict(train_features.Row(e))
			copy(train.Row(e), pred)
		}
		err = train.ToVector().Sub(train_labels.ToVector()).Norm(2)
		result.SetElem(i, 0, err/(float64(16)))
		err = predict.ToVector().Sub(test_labels.ToVector()).Norm(2)
		result.SetElem(i, 1, err/math.Sqrt(float64(101)))
	}

	// momentum
	n.InitWeight(w)
	params["batchSize"] = 1.0
	params["momentum"] = .93750
	start = time.Now()
	n.Train(train_features, train_labels, params)
	result.SetElem(0, 5, time.Since(start).Seconds())
	for e := 0; e < test_features.Rows(); e++ {
		pred := n.Predict(test_features.Row(e))
		copy(predict.Row(e), pred)
	}
	for e := 0; e < train_features.Rows(); e++ {
		pred := n.Predict(train_features.Row(e))
		copy(train.Row(e), pred)
	}
	err = train.ToVector().Sub(train_labels.ToVector()).Norm(2)
	result.SetElem(0, 3, err/(float64(16)))
	err = predict.ToVector().Sub(test_labels.ToVector()).Norm(2)
	result.SetElem(0, 4, err/math.Sqrt(float64(101)))

	for i := 1; i < maxrun; i++ {
		start = time.Now()
		n.Train(train_features, train_labels, params)
		result.SetElem(i, 5, time.Since(start).Seconds()+result.GetElem(i-1, 5))
		for e := 0; e < test_features.Rows(); e++ {
			pred := n.Predict(test_features.Row(e))
			copy(predict.Row(e), pred)
		}
		for e := 0; e < train_features.Rows(); e++ {
			pred := n.Predict(train_features.Row(e))
			copy(train.Row(e), pred)
		}
		err = train.ToVector().Sub(train_labels.ToVector()).Norm(2)
		result.SetElem(i, 3, err/(float64(16)))
		err = predict.ToVector().Sub(test_labels.ToVector()).Norm(2)
		result.SetElem(i, 4, err/math.Sqrt(float64(101)))
	}
	fmt.Printf("error is %v\n", result)
	result.SaveARFF("/tmp/batch_momentum.dat", false)
}

func main() {
	params := make(map[string]float64)
	params["learningRate"] = 1e-3
	params["seed"] = 2192018
	params["tol"] = 1e-2
	fmt.Printf("learning rate = %e\n", params["learningRate"])
	overfitting(params)
	fmt.Println("train with momentum .9375 and batchSize 16")
	batch_momentum(params)
}
