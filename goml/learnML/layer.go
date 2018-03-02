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
type Dims []int

const (
	LayerLinear LayerType = iota
	LayerTanh
	LayerConv
	LayerLeakyRectifier
	LayerMaxPooling2D
	LayerComposite
)

type Layer interface {
	Activate(x *matrix.Vector) *matrix.Vector
	BackProp(prevBlame *matrix.Vector)
	init(dim Dims, dims ...Dims)
	OutDim() Dims
	Wrap(activation, blame matrix.Vector, weight ...matrix.Vector) Layer
	Name() string
	UpdateGradient(in, gradient *matrix.Vector)
	Activation() *matrix.Vector
	Blame() *matrix.Vector
	Weight() *matrix.Vector
}

func NewLayer(t LayerType, dim Dims, dims ...Dims) Layer {
	var l Layer
	switch t {
	case LayerTanh:
		l = &layerTanh{}
	case LayerLeakyRectifier:
		l = &layerLeakyRectifier{}
	case LayerLinear:
		l = &layerLinear{}
	case LayerConv:
		l = &layerConv{}
	case LayerMaxPooling2D:
		l = &layerMaxPooling2D{}
	default:
		panic("Unsupported layer type!!!")
	}
	l.init(dim, dims...)
	return l
}

// layer implements identity activation function.
type layer struct {
	activation matrix.Vector
	blame      matrix.Vector
	weight     matrix.Vector
}

func (l *layer) OutDim() Dims {
	return Dims{len(l.activation)}
}

// dim = [out, inDim, innerDim]. Every layer must have an output
// dimension. layerLinear must have an input dimension.
func (l *layer) init(dim Dims, dims ...Dims) {
	l.activation = make(matrix.Vector, dim[0])
	l.blame = make(matrix.Vector, dim[0])
	l.weight = make(matrix.Vector, 0)
}

// layer.Activate returns the input.
func (l *layer) Activate(x *matrix.Vector) *matrix.Vector {
	panic("layer: Activate: not implemented!")
}

func (l *layer) BackProp(prevBlame *matrix.Vector) {
	panic("layer: BackProp: not implemented!")
}

// Wrap wraps a Layer around an activation Vector.
func (l *layer) Wrap(activation, blame matrix.Vector, weight ...matrix.Vector) Layer {
	panic("layer: Wrap: not implemented!")
}

func (l *layer) Name() string {
	panic("layer: Name: not implemented!")
}

func (l *layer) UpdateGradient(in, gradient *matrix.Vector) {}

func (l *layer) Activation() *matrix.Vector {
	return &(l.activation)
}

func (l *layer) Blame() *matrix.Vector {
	return &(l.blame)
}

func (l *layer) Weight() *matrix.Vector {
	return &(l.weight)
}
