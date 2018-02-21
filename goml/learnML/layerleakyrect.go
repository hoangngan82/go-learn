// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
)

type layerLeakyRectifier struct {
	layer
}

func (l *layerLeakyRectifier) Activate(x *matrix.Vector) *matrix.Vector {
	if len(l.layer.activation) != len(*x) {
		l.layer.activation = matrix.NewVector(len(*x), nil)
	}
	for i := 0; i < len(*x); i++ {
		if (*x)[i] < 0 {
			l.layer.activation[i] = 0.01 * (*x)[i]
		} else {
			l.layer.activation[i] = (*x)[i]
		}
	}
	return &(l.layer.activation)
}

func (l *layerLeakyRectifier) BackProp(prevBlame *matrix.Vector) {
	if len(*prevBlame) != len(l.layer.activation) {
		*prevBlame = matrix.NewVector(len(l.layer.activation), nil)
	}
	v := *prevBlame
	for i := 0; i < len(v); i++ {
		if l.layer.activation[i] < 0 {
			v[i] = .01
		} else {
			v[i] = 1.0
		}
	}
}

// Wrap wraps a Layer around an activation Vector.
func (l *layerLeakyRectifier) Wrap(activation matrix.Vector) Layer {
	matrix.Require(len(activation) == len(l.activation),
		"layer: Wrap: require len(activation) == len(l.activation)")
	var c layerLeakyRectifier
	c.layer.activation = activation
	//copy(c.activation, l.activation)
	return &c
}

func (l *layerLeakyRectifier) Name() string {
	return "Layer Leaky Rectifier"
}
