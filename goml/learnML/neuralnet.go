package learnML

import (
	"../matrix"
)

type NeuralNet struct {
	layers  []Layer
	weights []matrix.Vector
}

func (n *NeuralNet) Initialize(numLayers int, layerType interface{}) {
	n.layers = make([]Layer, numLayers)
	n.weights = make([]matrix.Vector, numLayers)
	switch layerType.(type) {
	case LayerLinear:
		for i := 0; i < numLayers; i++ {
			var v LayerLinear
			n.layers[i] = &v
		}
	default:
		panic("Only learnML.LayerLinear is a supported Layer")
	}
}

func (n *NeuralNet) Name() string {
	return "NeuralNet"
}

// Train trains the model and stores weights in weights[0].
func (n *NeuralNet) Train(features *matrix.Matrix, labels *matrix.Matrix) {
	var layerType interface{}
	layerType = n.layers[0]
	switch layerType.(type) {
	case *LayerLinear:
		n.weights[0] = n.layers[0].(*LayerLinear).OLS(features, labels)
	default:
		panic("Currently, only LayerLinear is supported")
	}
}

func (n *NeuralNet) TrainIncremental(feat matrix.Vector, lab matrix.Vector) {
	panic("not implemented")
}

// Predict will call Layer.Activate to compute predictions.
func (n *NeuralNet) Predict(in matrix.Vector) matrix.Vector {
	return n.layers[0].Activate(n.weights[0], in)
}

func (n *NeuralNet) FilterData(featIn *matrix.Matrix, labIn *matrix.Matrix, featOut *matrix.Matrix, labOut *matrix.Matrix) {
	panic("not implemented")
}
