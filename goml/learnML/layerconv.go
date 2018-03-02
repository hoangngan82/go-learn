package learnML

import (
	"../matrix"
)

type layerConv struct {
	layer
	in     Dims
	filter Dims
	out    Dims
}

func (l *layerConv) init(dim Dims, dims ...Dims) {
	inDim := len(dim)
	filterDim := len(dims[0])
	outDim := len(dims[1])
	matrix.Require(inDim+1 == outDim && outDim == filterDim,
		"layerconv: init:\nrequire %s\nbut get %d = %d && %d = %d\n",
		"inDim+1 == outDim && outDim == filterDim",
		inDim+1, outDim, outDim, filterDim)
	matrix.Require(dims[0][outDim-1] == dims[1][outDim-1],
		"layerconv: init:\nrequire %s\nbut get %d = %d\n",
		"filter[end] == out[end]", dims[0][outDim-1], dims[1][outDim-1])
	l.in = make(Dims, inDim)
	l.filter = make(Dims, filterDim)
	l.out = make(Dims, outDim)
	copy(l.in, dim)
	copy(l.filter, dims[0])
	copy(l.out, dims[1])

	// Outputs are stored in activation as usual.
	size := 1
	for i := 0; i < len(l.out); i++ {
		size *= l.out[i]
	}
	l.layer.activation = matrix.NewVector(size, nil)
	l.layer.blame = matrix.NewVector(size, nil)

	// weights are filters
	size = 1
	for i := 0; i < len(l.filter); i++ {
		size *= l.filter[i]
	}
	l.layer.weight = matrix.NewVector(size, nil)
}

func (l *layerConv) Activate(x *matrix.Vector) *matrix.Vector {
	in := matrix.NewTensor(*x, l.in)
	dc := len(l.in)

	sizeFilter := 1
	for i := 0; i < dc; i++ {
		sizeFilter *= l.filter[i]
	}

	sizeOut := 1
	for i := 0; i < dc; i++ {
		sizeOut *= l.out[i]
	}

	// reset l.layer.activation
	l.layer.activation.Fill(0.0)

	// Do convolution
	var out, filter *matrix.Tensor
	for i := 0; i < l.out[dc]; i++ {
		out = matrix.NewTensor(l.layer.activation[i*sizeOut:(i+1)*sizeOut], l.out[:dc])
		filter = matrix.NewTensor(l.layer.weight[i*sizeFilter:(i+1)*sizeFilter], l.filter[:dc])
		matrix.Convolve(in, filter, out, false, 1)
	}
	return &(l.layer.activation)
}

// BackProp compute Convolve(blame, weight, prevBlame)
func (l *layerConv) BackProp(prevBlame *matrix.Vector) {
	var in, out, filter *matrix.Tensor
	dc := len(l.in)
	out = matrix.NewTensor(*prevBlame, l.in)
	(*prevBlame).Fill(0.0)

	sizeFilter := 1
	for i := 0; i < dc; i++ {
		sizeFilter *= l.filter[i]
	}

	sizeIn := 1
	for i := 0; i < dc; i++ {
		sizeIn *= l.out[i]
	}

	// Do (backward) convolution
	for i := 0; i < l.out[dc]; i++ {
		in = matrix.NewTensor(l.layer.blame[i*sizeIn:(i+1)*sizeIn], l.out[:dc])
		filter = matrix.NewTensor(l.layer.weight[i*sizeFilter:(i+1)*sizeFilter], l.filter[:dc])
		matrix.Convolve(in, filter, out, true, 1)
	}
}

// gradient = in*blame
// Note that gradient and in are in the same dimensional space. blame
// is fo the same size as activation. Just as activation = in*weight,
// we do the same thing here (weight) = in*(activation).
func (l *layerConv) UpdateGradient(in *matrix.Vector, gradient *matrix.Vector) {
	var blame, prevActivation, gt *matrix.Tensor
	dc := len(l.in)
	prevActivation = matrix.NewTensor(*in, l.in)

	sizeGrad := 1
	for i := 0; i < dc; i++ {
		sizeGrad *= l.filter[i]
	}

	sizeBlame := 1
	for i := 0; i < dc; i++ {
		sizeBlame *= l.out[i]
	}

	// Do (backward) convolution
	for i := 0; i < l.out[dc]; i++ {
		blame = matrix.NewTensor(l.layer.blame[i*sizeBlame:(i+1)*sizeBlame], l.out[:dc])
		gt = matrix.NewTensor((*gradient)[i*sizeGrad:(i+1)*sizeGrad], l.filter[:dc])
		matrix.Convolve(prevActivation, blame, gt, false, 1)
	}
}

func (l *layerConv) Name() string {
	return "Layer Convolution"
}
