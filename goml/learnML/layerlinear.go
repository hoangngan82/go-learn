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
	activation matrix.Vector
	blame      matrix.Vector
	weight     matrix.Vector
}

func NewLinearLayer(out, in int, weight matrix.Vector) *layerLinear {
	size := (in + 1) * out
	matrix.Require(len(weight) == 0 || len(weight) == size,
		"NewLinearLayer: require len(weight) == 0 || len(weight) == size")
	l := layerLinear{}
	l.activation = matrix.NewVector(out, nil)
	l.blame = matrix.NewVector(out, nil)
	if len(weight) == 0 {
		l.weight = matrix.NewVector(size, nil)
	} else {
		l.weight = weight
	}
	return &l
}

// dim = [out[0], inDim, innerDim]. Every layer must have an output
// dimension. layerLinear must have an input dimension.
func (l *layerLinear) init(out, in int) {
	l.activation = matrix.NewVector(out, nil)
	l.blame = matrix.NewVector(out, nil)
	l.weight = matrix.NewVector((in+1)*out, nil)
}

func (l *layerLinear) Activate(xo *matrix.Vector) *matrix.Vector {
	x := matrix.NewVector(len(*xo)+1, nil)
	rows := len(x)
	copy(x, *xo)
	x[len(*xo)] = 1.0
	cols := len(l.weight) / rows
	xx := mat.NewDense(1, len(x), x)
	mm := mat.NewDense(rows, cols, l.weight)
	ac := mat.NewDense(1, cols, l.activation)
	ac.Mul(xx, mm)
	//M := matrix.NewMatrix(rows, cols, l.weight)
	//l.activation.ToMatrix().Mul(x.ToMatrix(), M, false, false)
	return &(l.activation)
}

// BackProp computes prevBlame = M^t*blame.
func (l *layerLinear) BackProp(prevBlame *matrix.Vector) {
	cols := len(l.activation)
	rows := len(l.weight) / cols
	for i := 0; i < rows-1; i++ {
		(*prevBlame)[i] = floats.Dot(l.blame, l.weight[i*cols:(i+1)*cols])
	}
	//var M *matrix.Matrix = &matrix.Matrix{}
	//M.WrapRows(matrix.NewMatrix(rows, cols, l.weight), []int{0}, []int{rows - 1})
	//(*prevBlame).ToMatrix().Mul(M, l.blame.ToMatrix(), false, true)
}

// Gradient is the derivative with respect to the weight. Thus, it
// has the same length as the weight.
func (l *layerLinear) UpdateGradient(in *matrix.Vector, gradient *matrix.Vector) {
	x := *in
	cols := len(l.blame)
	rows := len(*gradient) / cols
	// compute M += blame.OuterProd(x)
	for i := 0; i < rows-1; i++ {
		temp := (*gradient)[i*cols : (i+1)*cols]
		floats.AddScaled(temp, x[i], l.blame)
		//for j := 0; j < cols; j++ {
		//temp[j] += l.blame[j] * x[i]
		//}
	}

	// compute b += blame
	i := rows - 1
	temp := (*gradient)[i*cols : (i+1)*cols]
	floats.Add(temp, l.blame)
	//for j := 0; j < cols; j++ {
	//temp[j] += l.blame[j]
	//}
}

func (l *layerLinear) OutDim() Dimension {
	return Dimension{len(l.activation)}
}

func (l *layerLinear) Blame() *matrix.Vector {
	return &(l.blame)
}
