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

func (l *layerLeakyRectifier) Copy(activation matrix.Vector) Layer {
	var c layerLeakyRectifier
	c.layer = *(l.layer.Copy(activation).(*layer))
	return &c
}

func (l *layerLeakyRectifier) Name() string {
	return "Layer Leaky Rectifier"
}
