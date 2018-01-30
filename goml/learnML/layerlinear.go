// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
)

type LayerLinear struct {
	layer
}

func (l *LayerLinear) Activate(weights, xo matrix.Vector) matrix.Vector {
	x := make(matrix.Vector, len(xo)+1)
	rows := len(x)
	copy(x, xo)
	x[len(xo)] = 1.0
	cols := len(weights) / rows
	M := matrix.NewMatrix(rows, cols, weights)
	//fmt.Printf("M is %v\n", M)
	if len(l.activation) == 0 {
		l.activation = matrix.NewVector(cols, nil)
	}
	l.activation.ToMatrix().Mul(x.ToMatrix(), M, false, false)
	//fmt.Printf("M after is %v\n", M)
	return l.activation
}

func (l *LayerLinear) String() string {
	return l.activation.String()
}

func (l *LayerLinear) OLS(features, labels *matrix.Matrix) matrix.Vector {
	x := matrix.NewMatrix(features.Rows(), features.Cols()+1, nil)
	y := matrix.NewMatrix(labels.Rows(), labels.Cols(), nil)
	start := []int{0}
	end := []int{x.Rows()}
	x.CopyRows(features, start, end)
	y.CopyRows(labels, start, end)
	x.FillCol(-1, 1)
	return x.LeastSquare(y).ToVector()
}
