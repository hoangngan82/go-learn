package matrix

//import (
//"fmt"
//)

type Tensor struct {
	data Vector
	dims []int
}

func NewTensor(v Vector, dims []int) *Tensor {
	tot := 1
	for i := 0; i < len(dims); i++ {
		tot *= dims[i]
	}
	Require(tot == len(v),
		"NewTensor: size mismatched: Tensor (%d) != Vector (%d)\n",
		tot, len(v))
	var t Tensor
	t.dims = make([]int, len(dims))
	copy(t.dims, dims)
	t.data = v
	return &t
}

func Convolve(in, filter, out *Tensor, flipFilter bool, stride int) {
	// Precompute some values
	dc := len(in.dims)
	Require(dc == len(filter.dims) && dc == len(out.dims),
		"tensor: Convolve: Expected tensors with the same number of dimensions")
	kinner := make([]int, 5*dc)
	kouter := kinner[dc:]
	stepInner := kouter[dc:]
	stepFilter := stepInner[dc:]
	stepOuter := stepFilter[dc:]

	// Compute step sizes
	stepInner[0] = 1
	stepFilter[0] = 1
	stepOuter[0] = 1
	for i := 1; i < dc; i++ {
		stepInner[i] = stepInner[i-1] * in.dims[i-1]
		stepFilter[i] = stepFilter[i-1] * filter.dims[i-1]
		stepOuter[i] = stepOuter[i-1] * out.dims[i-1]
	}
	filterTail := stepFilter[dc-1]*filter.dims[dc-1] - 1

	// Do convolution
	var op, ip, fp int
	var padding, min, adj int
	for i := 0; i < dc; i++ {
		kouter[i] = 0
		kinner[i] = 0
		padding = (stride*(out.dims[i]-1) + filter.dims[i] - in.dims[i]) / 2
		min = padding
		if min > kouter[i] {
			min = kouter[i]
		}
		adj = (padding - min) - kinner[i]
		kinner[i] += adj
		fp += adj * stepFilter[i]
	}
	for {
		val := 0.0

		// Fix up the initial kinner positions
		for i := 0; i < dc; i++ {
			padding = (stride*(out.dims[i]-1) + filter.dims[i] - in.dims[i]) / 2
			min = padding
			if min > kouter[i] {
				min = kouter[i]
			}
			adj = (padding - min) - kinner[i]
			kinner[i] += adj
			fp += adj * stepFilter[i]
			ip += adj * stepInner[i]
		}
		for {
			if flipFilter {
				val += in.data[ip] * filter.data[filterTail-fp]
			} else {
				val += in.data[ip] * filter.data[fp]
			}

			// increment the kinner position
			var i int
			for i = 0; i < dc; i++ {
				kinner[i]++
				ip += stepInner[i]
				fp += stepFilter[i]
				padding = (stride*(out.dims[i]-1) + filter.dims[i] - in.dims[i]) / 2
				if kinner[i] < filter.dims[i] && kouter[i]+kinner[i]-padding < in.dims[i] {
					break
				}
				min = padding
				if min > kouter[i] {
					min = kouter[i]
				}
				adj = (padding - min) - kinner[i]
				kinner[i] += adj
				fp += adj * stepFilter[i]
				ip += adj * stepInner[i]
			}
			if i >= dc {
				break
			}
		}
		out.data[op] += val

		// increment the kouter position
		var i int
		for i = 0; i < dc; i++ {
			kouter[i]++
			op += stepOuter[i]
			ip += stride * stepInner[i]
			if kouter[i] < out.dims[i] {
				break
			}
			op -= kouter[i] * stepOuter[i]
			ip -= kouter[i] * stride * stepInner[i]
			kouter[i] = 0
		}
		if i >= dc {
			break
		}
	}
}
