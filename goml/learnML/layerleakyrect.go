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
	//if len(l.layer.activation) != len(*x) {
	//l.layer.activation = matrix.NewVector(len(*x), nil)
	//}
	copy(l.layer.activation, *x)
	for i := 0; i < len(*x); i++ {
		if (*x)[i] < 0 {
			l.layer.activation[i] = 0.01 * (*x)[i]
		} //else {
		//l.layer.activation[i] = (*x)[i]
		//}
	}
	return &(l.layer.activation)
}

func (l *layerLeakyRectifier) BackProp(prevBlame *matrix.Vector) {
	v := *prevBlame
	copy(v, l.layer.blame)
	for i := 0; i < len(v); i++ {
		if l.layer.activation[i] < 0 {
			v[i] = .01 * l.layer.blame[i]
		} //else {
		//v[i] = l.layer.blame[i]
		//}
	}
}

func (l *layerLeakyRectifier) Name() string {
	return "Layer Leaky Rectifier"
}
