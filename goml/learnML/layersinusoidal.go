package learnML

import (
	"../matrix"
	"math"
)

type layerSinusoidal struct {
	layer
	numSin     int
	derivative matrix.Vector
}

func (l *layerSinusoidal) init(dim Dims, dims ...Dims) {
	l.numSin = dim[0]
	n := l.numSin
	if len(dim) > 1 {
		n += dim[1]
	}
	l.layer.activation = make(matrix.Vector, n)
	l.layer.blame = make(matrix.Vector, n)
	// weight contains the derivatives of the activation function which
	// is needed in backprop.
	l.derivative = make(matrix.Vector, l.numSin)
}

func (l *layerSinusoidal) Activate(x *matrix.Vector) *matrix.Vector {
	for i := 0; i < l.numSin; i++ {
		l.layer.activation[i] = math.Sin((*x)[i])
		l.derivative[i] = math.Cos((*x)[i])
	}
	copy(l.layer.activation[l.numSin:], (*x)[l.numSin:])
	return &(l.layer.activation)
}

func (l *layerSinusoidal) BackProp(prevBlame *matrix.Vector) {
	v := *prevBlame
	for i := 0; i < l.numSin; i++ {
		v[i] = l.layer.blame[i] * l.derivative[i]
	}
	copy(v[l.numSin:], l.layer.blame[l.numSin:])
}

func (l *layerSinusoidal) Name() string {
	return "Layer Sinusoidal"
}
