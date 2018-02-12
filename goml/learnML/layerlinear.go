// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
)

type layerLinear struct {
	layer
	weight matrix.Vector
}

// dim = [out[0], inDim, innerDim]. Every layer must have an output
// dimension. layerLinear must have an input dimension.
func (l *layerLinear) Init(out Dimension, dim ...Dimension) {
	matrix.Require(len(dim) > 0, "layerLinear: Init must have two arguments\n")
	inDim := dim[0][0]
	l.layer.Init(out)
	l.weight = matrix.NewVector((inDim+1)*out[0], nil)
}

/*
[============ Begin of implementation for the Layer interface
*/

func (l *layerLinear) Activate(xo *matrix.Vector) *matrix.Vector {
	x := matrix.NewVector(len(*xo)+1, nil)
	rows := len(x)
	copy(x, *xo)
	x[len(*xo)] = 1.0
	cols := len(l.weight) / rows
	M := matrix.NewMatrix(rows, cols, l.weight)
	l.layer.activation.ToMatrix().Mul(x.ToMatrix(), M, false, false)
	return &(l.layer.activation)
}

// BackProp computes prevBlame = M^t*blame.
func (l *layerLinear) BackProp(prevBlame *matrix.Vector) {
	cols := len(l.layer.activation)
	rows := len(l.weight) / cols
	var M *matrix.Matrix = &matrix.Matrix{}
	M.WrapRows(matrix.NewMatrix(rows, cols, l.weight), []int{0}, []int{rows - 1})
	(*prevBlame).ToMatrix().Mul(M, l.layer.blame.ToMatrix(), false, true)
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
		for j := 0; j < cols; j++ {
			temp[j] += l.layer.blame[j] * x[i]
		}
	}

	// compute b += blame
	i := rows - 1
	temp := (*gradient)[i*cols : (i+1)*cols]
	for j := 0; j < cols; j++ {
		temp[j] += l.layer.blame[j]
	}
}

func (l *layerLinear) Activation() *matrix.Vector {
	return &(l.layer.activation)
}

func (l *layerLinear) Blame() *matrix.Vector {
	return &(l.layer.blame)
}

func (l *layerLinear) Name() string {
	return "Layer Linear"
}

/*
]============ End of implementation for the Layer interface
*/
