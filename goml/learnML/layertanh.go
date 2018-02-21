// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
	"math"
)

type layerTanh struct {
	layer
}

func (l *layerTanh) Activate(x *matrix.Vector) *matrix.Vector {
	if len(l.layer.activation) != len(*x) {
		l.layer.activation = matrix.NewVector(len(*x), nil)
	}
	for i := 0; i < len(*x); i++ {
		l.layer.activation[i] = math.Tanh((*x)[i])
	}
	return &(l.layer.activation)
}

func (l *layerTanh) BackProp(prevBlame *matrix.Vector) {
	if len(*prevBlame) != len(l.layer.activation) {
		*prevBlame = matrix.NewVector(len(l.layer.activation), nil)
	}
	v := *prevBlame
	for i := 0; i < len(v); i++ {
		v[i] = 1.0 - l.layer.activation[i]*l.layer.activation[i]
	}
}

func (l *layerTanh) Name() string {
	return "Layer Tanh"
}

// Wrap wraps a Layer around an activation Vector.
func (l *layerTanh) Wrap(activation matrix.Vector) Layer {
	matrix.Require(len(activation) == len(l.activation),
		"layer: Wrap: require len(activation) == len(l.activation)")
	var c layerTanh
	c.layer.activation = activation
	//copy(c.activation, l.activation)
	return &c
}
