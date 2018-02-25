// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
	"gonum.org/v1/gonum/floats"
	"math"
)

type layerTanh struct {
	layer
}

func (l *layerTanh) Activate(x *matrix.Vector) *matrix.Vector {
	//if len(l.layer.activation) != len(*x) {
	//l.layer.activation = matrix.NewVector(len(*x), nil)
	//}
	for i := 0; i < len(*x); i++ {
		l.layer.activation[i] = math.Tanh((*x)[i])
	}
	return &(l.layer.activation)
}

func (l *layerTanh) BackProp(prevBlame *matrix.Vector) {
	v := *prevBlame
	floats.MulTo(v, l.layer.activation, l.layer.activation)
	floats.Mul(v, l.layer.blame)
	floats.SubTo(v, l.layer.blame, v)
}

func (l *layerTanh) Name() string {
	return "Layer Tanh"
}

// Wrap wraps a Layer around an activation Vector.
func (l *layerTanh) Wrap(activation, blame matrix.Vector, weight ...matrix.Vector) Layer {
	matrix.Require(len(activation) == len(blame),
		"layerTanh: Wrap: require len(activation) == len(blame)")
	var c layerTanh
	c.layer.activation = activation
	c.layer.blame = blame
	return &c
}
