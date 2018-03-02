package learnML

import (
	"../matrix"
)

type Imputer struct {
	mode matrix.Vector
}

func (im *Imputer) Train(data matrix.Matrix) {
	im.mode = matrix.NewVector(data.Cols(), nil)
	for i := 0; i < len(im.mode); i++ {
		if data.ValueCount(i) == 0 {
			im.mode[i] = data.ColumnMean(i)
		} else {
			im.mode[i] = float64(data.MostCommonValue(i))
		}
	}
}

func (im *Imputer) Transform(in, out *matrix.Vector) {
	matrix.Require(len(im.mode) == len(*in),
		"Imputer: Transform: require in vector of size %d but get %d\n",
		len(im.mode), len(*in))
	if len(*out) != len(*in) {
		*out = matrix.NewVector(len(*in), nil)
	}
	im.transform(in, out)
}

func (im *Imputer) transform(in, out *matrix.Vector) {
	for i := 0; i < len(*in); i++ {
		if (*in)[i] == matrix.UNKNOWN_VALUE {
			(*out)[i] = im.mode[i]
		} else {
			(*out)[i] = (*in)[i]
		}
	}
}

func (b *Imputer) TransformBatch(input, output *matrix.Matrix) {
	in := *input
	out := *output
	matrix.Require(in.Cols() == len(b.mode), "%s %d %s %d\n",
		"Imputer: TransformBatch: Expected input matrix of size *x",
		len(b.mode), ", got *x", in.Cols())
	if out.Rows() != in.Rows() || out.Cols() != in.Cols() {
		*output = *(matrix.NewMatrix(in.Rows(), in.Cols(), nil))
		out = *output
	}
	output.CopyMetadata(input)
	var inVec, outVec matrix.Vector
	for i := 0; i < in.Rows(); i++ {
		inVec = in.Row(i)
		outVec = out.Row(i)
		b.transform(&inVec, &outVec)
	}
}

func (im *Imputer) Untransform(in, out *matrix.Vector) {
	panic("Imputer: Untransform: not implemented!")
}
