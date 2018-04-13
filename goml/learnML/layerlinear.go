// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
// LayerLinear is not a "layer". All layers in this code represent
// activation functions.
package learnML

import (
	"../matrix"
	//"gonum.org/v1/gonum/floats"
	//"gonum.org/v1/gonum/mat"
)

type layerLinear struct {
	layer
	l1, l2 float64 // li is for Li-regularization.
}

// initialize a layerLinear
// li is passed as the ratio of two integers. For example, with
// init([]int{}, []int{3, 2}, []int(1, 100)), we have
// l1 = 3/2 = 1.5 and l2 = 1/100 = .01
func (l *layerLinear) init(dim Dims, dims ...Dims) {
	matrix.Require(len(dim) == 2, "layerLinear: init: require dim = 2")
	out := dim[1]
	in := dim[0]
	l.layer.activation = matrix.NewVector(out, nil)
	l.layer.blame = matrix.NewVector(out, nil)
	l.layer.weight = matrix.NewVector((in+1)*out, nil)
	if len(dims) > 0 { // L1-regularization
		l.l1 = float64(dims[0][0]) / float64(dims[0][1])
	}
	if len(dims) > 1 { // L2-regularization
		l.l2 = float64(dims[1][0]) / float64(dims[1][1])
	}
}

func (l *layerLinear) Activate(xo *matrix.Vector) *matrix.Vector {
	x := *xo
	rows := len(x)
	cols := len(l.layer.weight) / (rows + 1)
	k := rows * cols
	w := l.layer.weight[k:]
	for i := 0; i < len(l.layer.activation); i++ {
		l.layer.activation[i] = w[i]
	}
	// use i for loop inside j for loop is much faster than the other
	// way around
	for j := 0; j < rows; j++ {
		k = j * cols
		w = l.layer.weight[k:]
		for i := 0; i < cols; i++ {
			l.layer.activation[i] += x[j] * w[i]
		}
	}
	// use go routines and channels in this case is slower than double
	// for loops
	//c := make(chan int, cols)
	//for k := 0; k < cols; k++ {
	//go func(i int, ch chan int) {
	//for j := 0; j < rows; j++ {
	//l.layer.activation[i] += x[j] * l.layer.weight[i+j*cols]
	//}
	//ch <- 1
	//}(k, c)
	//<-c
	//}
	// use floats routines to vectorize is close to gonum mat mul
	//for i := 0; i < rows; i++ {
	//floats.AddScaled(l.layer.activation, x[i],
	//l.layer.weight[i*cols:(i+1)*cols])
	//}
	// use gonum mat mul is the fastest
	//xx := mat.NewDense(1, len(x), x)
	//mm := mat.NewDense(rows, cols, l.layer.weight)
	//ac := mat.NewDense(1, cols, l.layer.activation)
	//ac.Mul(xx, mm)
	return &(l.layer.activation)
}

// BackProp computes prevBlame = M^t*blame.
func (l *layerLinear) BackProp(prevBlame *matrix.Vector) {
	cols := len(l.layer.activation)
	rows := len(l.layer.weight) / cols
	vb := matrix.NewVector(cols, l.layer.blame)
	for i := 0; i < rows-1; i++ {
		vw := matrix.NewVector(cols, l.layer.weight[i*cols:(i+1)*cols])
		(*prevBlame)[i] = vb.Dot(vw)
		//(*prevBlame)[i] = floats.Dot(l.layer.blame, l.layer.weight[i*cols:(i+1)*cols])
	}
}

// Gradient is the derivative with respect to the weight. Thus, it
// has the same length as the weight.
func (l *layerLinear) UpdateGradient(in *matrix.Vector, gradient *matrix.Vector) {
	x := *in
	cols := len(l.layer.blame)
	rows := len(*gradient) / cols

	l1 := float64((rows - 1) * cols)
	l2 := l.l2 / l1
	l1 = l.l1 / l1
	// compute M += blame.OuterProd(x)
	bb := l.layer.blame
	for i := 0; i < rows-1; i++ {
		temp := (*gradient)[i*cols : (i+1)*cols]
		w := l.layer.weight[i*cols : (i+1)*cols]
		for j := 0; j < cols; j++ {
			temp[j] += x[i]*bb[j] + l2*w[j]
			if w[j] < 0 {
				temp[j] -= l1
			} else {
				temp[j] += l1
			}
		}
		//floats.AddScaled(temp, x[i], l.layer.blame)
	}

	// compute b += blame
	i := rows - 1
	temp := (*gradient)[i*cols : (i+1)*cols]
	for i := 0; i < cols; i++ {
		temp[i] += bb[i]
	}
	//floats.Add(temp, l.layer.blame)
}

func (l *layerLinear) Name() string {
	return "Layer Linear"
}
