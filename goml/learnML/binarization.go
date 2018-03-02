package learnML

import (
	"../matrix"
)

type Binarization struct {
	vals      []int
	totalVals int
}

func (b *Binarization) Train(data matrix.Matrix) {
	b.totalVals = 0
	b.vals = make([]int, data.Cols())
	for i := 0; i < data.Cols(); i++ {
		n := data.ValueCount(i)
		if n < 3 {
			n = 1
		}
		b.vals[i] = n
		b.totalVals += n
	}
}

func (b *Binarization) Transform(inVec *matrix.Vector, outVec *matrix.Vector) {
	matrix.Require(len(*inVec) == len(b.vals), "%s %d %s %d",
		"Binarization: Transform received unexpected in-vector size. Expected",
		len(b.vals), ", got ", len(*inVec))
	if len(*outVec) != b.totalVals {
		*outVec = matrix.NewVector(b.totalVals, nil)
	}
	b.transform(inVec, outVec)
}

func (b *Binarization) transform(inVec *matrix.Vector, outVec *matrix.Vector) {
	in := *inVec
	out := *outVec

	id := 0
	out.Fill(0.0)
	for i := 0; i < len(in); i++ {
		if b.vals[i] == 1 {
			out[id] = in[i]
			id++
		} else {
			if in[i] != matrix.UNKNOWN_VALUE {
				matrix.Require(int(in[i]) < b.vals[i], "%s%d%s %d\n",
					"Binarization: Transform: Value out of range. Expected [0-",
					b.vals[i]-1, "], got ", int(in[i]))
				out[id+int(in[i])] = 1.0
				id += b.vals[i]
			}
		}
	}
}

func (b *Binarization) Untransform(inVec *matrix.Vector, outVec *matrix.Vector) {
	matrix.Require(len(*inVec) == b.totalVals, "%s %s %d %s %d\n",
		"Binarization: Untransform received unexpected in-vector size.",
		"Expected", b.totalVals, ", got", len(*inVec))

	if len(*outVec) != len(b.vals) {
		*outVec = matrix.NewVector(len(b.vals), nil)
	}
	in := *inVec
	out := *outVec
	out.Fill(0.0)
	id := 0
	for i := 0; i < len(out); i++ {
		if b.vals[i] == 1 {
			out[i] = in[id]
			id++
		} else {
			maxId := 0
			maxVal := in[id]
			for j := 1; j < b.vals[i]; j++ {
				id++
				if in[id] > maxVal {
					maxVal = in[id]
					maxId = j
				}
			}
			out[i] = float64(maxId)
		}
	}
}

func (b *Binarization) TransformBatch(input, output *matrix.Matrix) {
	in := *input
	out := *output
	matrix.Require(in.Cols() == len(b.vals), "%s %d %s %d\n",
		"Binarization: TransformBatch: Expected input matrix of size *x",
		len(b.vals), ", got *x", in.Cols())
	if out.Rows() != in.Rows() || out.Cols() != b.totalVals {
		*output = *(matrix.NewMatrix(in.Rows(), b.totalVals, nil))
		out = *output
	}
	var inVec, outVec matrix.Vector
	for i := 0; i < in.Rows(); i++ {
		inVec = in.Row(i)
		outVec = out.Row(i)
		b.transform(&inVec, &outVec)
	}
}
