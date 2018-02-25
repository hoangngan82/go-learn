// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
// LayerLinear is not a "layer". All layers in this code represent
// activation functions.
package learnML

import (
	"../matrix"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type layerLinear struct {
	layer
}

// initialize a layerLinear
func (l *layerLinear) init(dim Dims, dims ...Dims) {
	matrix.Require(len(dim) == 2, "layerLinear: init: require dim = 2")
	out := dim[1]
	in := dim[0]
	l.layer.activation = matrix.NewVector(out, nil)
	l.layer.blame = matrix.NewVector(out, nil)
	l.layer.weight = matrix.NewVector((in+1)*out, nil)
}

func (l *layerLinear) Activate(xo *matrix.Vector) *matrix.Vector {
	x := matrix.NewVector(len(*xo)+1, nil)
	rows := len(x)
	copy(x, *xo)
	x[len(*xo)] = 1.0
	cols := len(l.layer.weight) / rows
	xx := mat.NewDense(1, len(x), x)
	mm := mat.NewDense(rows, cols, l.layer.weight)
	ac := mat.NewDense(1, cols, l.layer.activation)
	ac.Mul(xx, mm)
	return &(l.layer.activation)
}

// BackProp computes prevBlame = M^t*blame.
func (l *layerLinear) BackProp(prevBlame *matrix.Vector) {
	cols := len(l.layer.activation)
	rows := len(l.layer.weight) / cols
	for i := 0; i < rows-1; i++ {
		(*prevBlame)[i] = floats.Dot(l.layer.blame, l.layer.weight[i*cols:(i+1)*cols])
	}
}

// Gradient is the derivative with respect to the weight. Thus, it
// has the same length as the weight.
func (l *layerLinear) UpdateGradient(in *matrix.Vector, gradient *matrix.Vector) {
	x := *in
	cols := len(l.layer.blame)
	rows := len(*gradient) / cols
	// compute M += blame.OuterProd(x)
	for i := 0; i < rows-1; i++ {
		temp := (*gradient)[i*cols : (i+1)*cols]
		floats.AddScaled(temp, x[i], l.layer.blame)
	}

	// compute b += blame
	i := rows - 1
	temp := (*gradient)[i*cols : (i+1)*cols]
	floats.Add(temp, l.layer.blame)
}

func (l *layerLinear) Name() string {
	return "Layer Linear"
}
