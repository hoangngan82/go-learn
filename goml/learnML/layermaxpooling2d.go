package learnML

import (
	"../matrix"
	//"fmt"
)

// We only use filter of size 2x2 and stride of size 2x2. This will
// reduce the size of the input by 4 (2 in each direction).
type layerMaxPooling2D struct {
	layer
	maxid []int
	out   Dims
}

func (l *layerMaxPooling2D) init(dim Dims, dims ...Dims) {
	size := 1
	for i := 0; i < len(dim); i++ {
		size *= dim[i]
	}
	size /= 4

	l.out = make(Dims, len(dim))
	copy(l.out, dim)
	l.out[0] /= 2
	l.out[1] /= 2

	l.layer.activation = matrix.NewVector(size, nil)
	l.layer.blame = matrix.NewVector(size, nil)
	l.maxid = make([]int, size)
}

func (l *layerMaxPooling2D) Activate(x *matrix.Vector) *matrix.Vector {
	N := 1
	for i := 2; i < len(l.out); i++ {
		N *= l.out[i]
	}

	size := l.out[0] * l.out[1]

	// Do 2D max pooling
	var inID, outID int
	for i := 0; i < N; i++ {
		// loop through the 2D slice of the output
		maxid := l.maxid[i*size : (i+1)*size]
		out := l.layer.activation[i*size : (i+1)*size]
		in := (*x)[i*4*size : (i+1)*4*size]
		for r := 0; r < l.out[0]; r++ {
			for c := 0; c < l.out[1]; c++ {
				outID = r*l.out[0] + c
				inID = (r<<2)*l.out[0] + (c << 1)
				maxid[outID] = inID
				out[outID] = in[inID]

				inID++
				if out[outID] < in[inID] {
					maxid[outID] = inID
					out[outID] = in[inID]
				}

				inID = ((r<<2)|2)*l.out[0] + (c << 1)
				if out[outID] < in[inID] {
					maxid[outID] = inID
					out[outID] = in[inID]
				}

				inID++
				if out[outID] < in[inID] {
					maxid[outID] = inID
					out[outID] = in[inID]
				}
			}
		}
	}

	return &(l.layer.activation)
}

func (l *layerMaxPooling2D) BackProp(prevBlame *matrix.Vector) {
	N := 1
	for i := 2; i < len(l.out); i++ {
		N *= l.out[i]
	}

	size := l.out[0] * l.out[1]

	// Do 2D max pooling
	var id int
	(*prevBlame).Fill(0.0)
	for i := 0; i < N; i++ {
		// loop through the 2D slice of the output
		in := l.layer.blame[i*size : (i+1)*size]
		maxid := l.maxid[i*size : (i+1)*size]
		out := (*prevBlame)[i*4*size : (i+1)*4*size]
		for r := 0; r < l.out[0]; r++ {
			for c := 0; c < l.out[1]; c++ {
				id = r*l.out[0] + c
				out[maxid[id]] = in[id]
			}
		}
	}
}

func (l *layerMaxPooling2D) Name() string {
	return "Layer Max Pooling 2D"
}
