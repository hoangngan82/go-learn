package learnML

import (
	"../matrix"
	"fmt"
)

type Normalizer struct {
	inputMin matrix.Vector
	inputMax matrix.Vector
}

func (n *Normalizer) Train(data matrix.Matrix) {
	N := data.Cols()
	n.inputMin = matrix.NewVector(N, nil)
	n.inputMax = matrix.NewVector(N, nil)
	for i := 0; i < N; i++ {
		fmt.Printf("at %5d: value count = %d\n", i, data.ValueCount(i))
		if data.ValueCount(i) == 0 {
			n.inputMin[i] = data.ColumnMin(i)
			n.inputMax[i] = data.ColumnMax(i)
			if n.inputMax[i] < n.inputMin[i]+1e-9 {
				n.inputMax[i] = n.inputMin[i] + 1e-9
			}
		} else {
			// Don't do nominal attributues
			n.inputMin[i] = matrix.UNKNOWN_VALUE
			n.inputMax[i] = matrix.UNKNOWN_VALUE
		}
	}
}

func (n *Normalizer) transform(inVec, outVec *matrix.Vector) {
	in := *inVec
	out := *outVec
	for i := 0; i < len(in); i++ {
		if n.inputMax[i] == matrix.UNKNOWN_VALUE {
			// Do nothing with nominal attribute
			out[i] = in[i]
		} else {
			out[i] = (in[i] - n.inputMin[i]) / (n.inputMax[i] - n.inputMin[i])
		}
	}
}

func (n *Normalizer) Transform(inVec, outVec *matrix.Vector) {
	matrix.Require(len(n.inputMin) == len(*inVec), "%s %d %s %d.\n",
		"Normalizer: Transform: Expected input vector of length",
		len(n.inputMax), "but get", len(*inVec))
	if len(*outVec) != len(*inVec) {
		*outVec = matrix.NewVector(len(*inVec), nil)
	}
	n.transform(inVec, outVec)
}

func (n *Normalizer) Untransform(inVec, outVec *matrix.Vector) {
	matrix.Require(len(n.inputMin) == len(*inVec), "%s %d %s %d.\n",
		"Normalizer: Transform: Expected input vector of length",
		len(n.inputMax), "but get", len(*inVec))
	if len(*outVec) != len(*inVec) {
		*outVec = matrix.NewVector(len(*inVec), nil)
	}
	in := *inVec
	out := *outVec
	for i := 0; i < len(in); i++ {
		if in[i] == matrix.UNKNOWN_VALUE {
			out[i] = matrix.UNKNOWN_VALUE
			continue
		}
		if n.inputMax[i] == matrix.UNKNOWN_VALUE {
			// Do nothing with nominal attribute
			out[i] = in[i]
		} else {
			if in[i] == matrix.UNKNOWN_VALUE {
				out[i] = matrix.UNKNOWN_VALUE
			} else {
				out[i] = in[i]*(n.inputMax[i]-n.inputMin[i]) + n.inputMin[i]
			}
		}
	}
}

func (n *Normalizer) TransformBatch(input, output *matrix.Matrix) {
	in := *input
	out := *output
	matrix.Require(in.Cols() == len(n.inputMax), "%s %d %s %d\n",
		"Normalizer: TransformBatch: Expected input matrix of size *x",
		len(n.inputMin), ", got *x", in.Cols())
	if out.Rows() != in.Rows() || out.Cols() != in.Cols() {
		*output = *(matrix.NewMatrix(in.Rows(), in.Cols(), nil))
		out = *output
	}
	output.CopyMetadata(input)
	var inVec, outVec matrix.Vector
	for i := 0; i < in.Rows(); i++ {
		inVec = in.Row(i)
		outVec = out.Row(i)
		n.transform(&inVec, &outVec)
	}
}
