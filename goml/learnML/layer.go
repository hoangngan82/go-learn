// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
// Only layerLinear has weight. Thus, we move weight to LayerLinear
// and because of that Activate and BackProp functions do not need
// the weight parameter. Also, LayerLinear's are special because it
// connects all other nonlinear layers. Since nonlinear layers does
// backprop different than LayerLinear, we will not consider
// LayerLinear of type Layer (which is activation function).
package learnML

import (
	"../matrix"
)

type LayerType int
type Dimension []int

const (
	LayerLinear LayerType = iota
	LayerTanh
	LayerConv
	LayerLeakyRectifier
	LayerMaxPooling2D
)

type Layer interface {
	Activate(x *matrix.Vector) *matrix.Vector
	BackProp(prevBlame *matrix.Vector)
	Activation() *matrix.Vector
	init(out Dimension, dim ...Dimension)
	OutDim() Dimension
	Copy(activation matrix.Vector) Layer
	Name() string
}

func NewLayer(t LayerType, out Dimension, dim ...Dimension) Layer {
	var l Layer
	switch t {
	case LayerTanh:
		l = &layerTanh{}
	case LayerLeakyRectifier:
		l = &layerLeakyRectifier{}
	case LayerLinear:
		l = &layer{}
	default:
		panic("Unsupported layer type!!!")
	}
	l.init(out, dim...)
	return l
}

// layer implements identity activation function.
type layer struct {
	activation matrix.Vector
}

func (l *layer) OutDim() Dimension {
	return Dimension{len(l.activation)}
}

// dim = [out, inDim, innerDim]. Every layer must have an output
// dimension. layerLinear must have an input dimension.
func (l *layer) init(out Dimension, dim ...Dimension) {
	l.activation = matrix.NewVector(out[0], nil)
}

// layer.Activate returns the input.
func (l *layer) Activate(x *matrix.Vector) *matrix.Vector {
	if len(l.activation) != len(*x) {
		l.activation = matrix.NewVector(len(*x), nil)
	}
	l.activation.Copy(*x)
	return x
}

// layer.BackProp returns the derivative of the identity activation
// function, which is 1.
func (l *layer) BackProp(prevBlame *matrix.Vector) {
	if len(*prevBlame) != len(l.activation) {
		*prevBlame = matrix.NewVector(len(l.activation), nil)
	}
	(*prevBlame).Fill(1.0)
}

func (l *layer) Activation() *matrix.Vector {
	return &(l.activation)
}

func (l *layer) Copy(activation matrix.Vector) Layer {
	matrix.Require(len(activation) == 0 || len(activation) == len(l.activation),
		"layer: Copy: require len(activation) == 0 || len(activation) == len(l.activation)")
	var c layer
	n := len(l.activation)
	if len(activation) == 0 {
		c.activation = make(matrix.Vector, n)
	} else {
		c.activation = activation
	}
	copy(c.activation, l.activation)
	return &c
}

func (l *layer) Name() string {
	return "Identity Layer"
}
