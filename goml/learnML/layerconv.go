package learnML

import (
	"../matrix"
)

type layerConv struct {
	layer
	in, filter, out Dims
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

	// Do convolution
	var out, filter *matrix.Tensor
	for i := 0; i < l.out[dc]; i++ {
		out = matrix.NewTensor(l.layer.activation[i*sizeOut:(i+1)*sizeOut], l.out[:dc])
		filter = matrix.NewTensor(l.layer.weight[i*sizeFilter:(i+1)*sizeFilter], l.filter[:dc])
		matrix.Convolve(in, filter, out, false, 1)
	}
	return &(l.layer.activation)
}

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
		in = matrix.NewTensor(l.layer.activation[i*sizeIn:(i+1)*sizeIn], l.out[:dc])
		filter = matrix.NewTensor(l.layer.weight[i*sizeFilter:(i+1)*sizeFilter], l.filter[:dc])
		matrix.Convolve(in, filter, out, true, 1)
	}
}

func (l *layerConv) Name() string {
	return "Layer Convolution"
}
