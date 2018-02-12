// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
// Only layerLinear has weight. Thus, we move weight to LayerLinear
// and because of that Activate and BackProp functions do not need
// the weight parameter.
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
)

type Layer interface {
	Activate(x *matrix.Vector) *matrix.Vector
	BackProp(prevBlame *matrix.Vector)
	UpdateGradient(x *matrix.Vector, gradient *matrix.Vector)
	Activation() *matrix.Vector
	Blame() *matrix.Vector
	Init(out Dimension, dim ...Dimension)
	OutDim() Dimension
	Copy(activation, blame matrix.Vector) Layer
	Name() string
}

func NewLayer(t LayerType, out Dimension, dim ...Dimension) Layer {
	var l Layer
	switch t {
	case LayerTanh:
		l = &layerTanh{}
	case LayerLinear:
		l = &layerLinear{}
	case LayerLeakyRectifier:
		l = &layerLeakyRectifier{}
	default:
		l = &layer{}
	}
	l.Init(out, dim...)
	return l
}

// layer implements identity activation function.
type layer struct {
	activation, blame matrix.Vector
}

func (l *layer) OutDim() Dimension {
	return Dimension{len(l.activation)}
}

// dim = [out, inDim, innerDim]. Every layer must have an output
// dimension. layerLinear must have an input dimension.
func (l *layer) Init(out Dimension, dim ...Dimension) {
	l.activation = matrix.NewVector(out[0], nil)
	l.blame = matrix.NewVector(out[0], nil)
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

func (l *layer) UpdateGradient(x, gradient *matrix.Vector) {
	panic("not implemented")
}

func (l *layer) Activation() *matrix.Vector {
	return &(l.activation)
}

func (l *layer) Blame() *matrix.Vector {
	return &(l.blame)
}

func (l *layer) Copy(activation, blame matrix.Vector) Layer {
	matrix.Require(len(activation) == 0 || len(activation) == len(l.activation),
		"layer: Copy: require len(activation) == 0 || len(activation) == len(l.activation)")
	matrix.Require(len(blame) == 0 || len(blame) == len(l.blame),
		"layer: Copy: require len(blame) == 0 || len(blame) == len(l.blame)")
	var c layer
	n := len(l.activation)
	if len(activation) == 0 {
		c.activation = make(matrix.Vector, n)
	} else {
		c.activation = activation
	}
	if len(blame) == 0 {
		c.blame = make(matrix.Vector, n)
	} else {
		c.blame = blame
	}
	copy(c.activation, l.activation)
	copy(c.blame, l.blame)
	return &c
}

func (l *layer) Name() string {
	return "Identity Layer"
}
