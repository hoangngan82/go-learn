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
	Activation() *matrix.Vector
	init(out Dims, dim ...Dims)
	OutDim() Dims
	Wrap(activation matrix.Vector) Layer
	Name() string
}

func NewLayer(t LayerType, out Dims, dim ...Dims) Layer {
	var l Layer
	switch t {
	case LayerTanh:
		l = &layerTanh{}
	case LayerLeakyRectifier:
		l = &layerLeakyRectifier{}
	case LayerLinear:
		l = &layer{}
	case LayerConv:
		l = &layerConv{}
	case LayerComposite:
		l = &layerComposite{}
	default:
		panic("Unsupported layer type!!!")
	}
	l.init(out, dim...)
	return l
}

// layer implements identity activation function.
type layer struct {
	activation matrix.Vector
	blame      matrix.Vector
}

func (l *layer) OutDim() Dims {
	return Dims{len(l.activation)}
}

// dim = [out, inDim, innerDim]. Every layer must have an output
// dimension. layerLinear must have an input dimension.
func (l *layer) init(out Dims, dim ...Dims) {
	panic("layer: init: not implemented!")
}

// layer.Activate returns the input.
func (l *layer) Activate(x *matrix.Vector) *matrix.Vector {
	panic("layer: Activate: not implemented!")
}

func (l *layer) BackProp(prevBlame *matrix.Vector) {
	panic("layer: BackProp: not implemented!")
}

// Wrap wraps a Layer around an activation Vector.
func (l *layer) Wrap(activation matrix.Vector) Layer {
	panic("layer: Wrap: not implemented!")
}

func (l *layer) Name() string {
	panic("layer: Name: not implemented!")
}
