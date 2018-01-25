package learnML

import (
	"fmt"
	str "strings"
)

type NeuralNet struct {
}

func (n *NeuralNet) Name() string {
	panic("not implemented")
}

func (n *NeuralNet) Train(features *matrix.Matrix, labels *matrix.Matrix) {
	panic("not implemented")
}

func (n *NeuralNet) TrainIncremental(feat learnML.Vector, lab learnML.Vector) {
	panic("not implemented")
}

func (n *NeuralNet) Predict(in learnML.Vector) learnML.Vector {
	panic("not implemented")
}

func (n *NeuralNet) CountMisclassifications(features *matrix.Matrix, labels *matrix.Matrix) {
	panic("not implemented")
}

func (n *NeuralNet) Filter_data(feat_in *matrix.Matrix, lab_in *matrix.Matrix, feat_out *matrix.Matrix, lab_out *matrix.Matrix) {
	panic("not implemented")
}
