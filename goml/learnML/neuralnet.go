package learnML

import (
	"fmt"
	str "strings"
)

type NeuralNet struct {
	layers []Layer
}

func (n *NeuralNet) Name() string {
	return "NeuralNet"
}

func (n *NeuralNet) Train(features *matrix.Matrix, labels *matrix.Matrix) {
	panic("not implemented")
}

func (n *NeuralNet) TrainIncremental(feat matrix.Vector, lab matrix.Vector) {
	panic("not implemented")
}

func (n *NeuralNet) Predict(in matrix.Vector) matrix.Vector {
	panic("not implemented")
}

func (n *NeuralNet) FilterData(featIn *matrix.Matrix, labIn *matrix.Matrix, featOut *matrix.Matrix, labOut *matrix.Matrix) {
	panic("not implemented")
}
