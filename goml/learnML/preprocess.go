package learnML

import (
	"../matrix"
)

type Preprocess interface {
	Train(data matrix.Matrix)
	Transform(in, out *matrix.Vector)
	Untransform(in, out *matrix.Vector)
}

func TransformBatch(p Preprocess, in, out *matrix.Matrix) {
	var inVec, outVec matrix.Vector
	for i := 0; i < in.Rows(); i++ {
		inVec = in.Row(i)
		outVec = out.Row(i)
		p.Transform(&inVec, &outVec)
	}
}
