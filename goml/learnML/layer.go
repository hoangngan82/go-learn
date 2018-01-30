// Package learnML contains all my answer to homework for the class
// ISYS 5063 - Machine Learning, taught by Michael Gashler at UARK,
// Fayetteville, AR.
package learnML

import (
	"../matrix"
)

type Layer interface {
	Activate(weights, x matrix.Vector) matrix.Vector
	BackProp(weights, prevBlame matrix.Vector)
	UpdateGradient(x, gradient matrix.Vector)
	Blame(weights, x matrix.Vector) matrix.Vector
}

type layer struct {
	activation, blame matrix.Vector
}

func (l *layer) Activate(weights matrix.Vector, x matrix.Vector) matrix.Vector {
	panic("not implemented")
}