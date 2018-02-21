package learnML

import (
	"../matrix"
	"gonum.org/v1/gonum/floats"
)

type layerStack struct {
	layer
	blame matrix.Vector
	unit  []Layer
}

func NewLayerStack(first Layer, next ...Layer) *layerStack {
	var l layerStack
	size := len(next) + 1
	l.unit = make([]Layer, size)

	l.layer.activation = matrix.NewVector(len(*(first.Activation())), nil)

	l.unit[0] = first.Wrap(l.layer.activation)
	for i := 0; i < len(next); i++ {
		j := i + 1
		l.unit[j] = next[i].Wrap(l.layer.activation)
	}
	l.blame = matrix.NewVector(len(l.layer.activation), nil)
	return &l
}

func (l *layerStack) Activate(x *matrix.Vector) *matrix.Vector {
	temp := matrix.NewVector(len(l.blame), nil)
	l.unit[0].Activate(x)
	l.unit[0].BackProp(&(l.blame))
	for i := 1; i < len(l.unit); i++ {
		l.unit[i].Activate(l.unit[i-1].Activation())
		l.unit[i].BackProp(&temp)
		floats.Mul(l.blame, temp)
	}
	return &(l.layer.activation)
}

func (l *layerStack) BackProp(prevBlame *matrix.Vector) {
	matrix.Require(len(*prevBlame) == len(l.blame),
		"LayerStack: BackProp: require len(*prevBlame) == len(l.blame)")
	copy(*prevBlame, l.blame)
}

func (l *layerStack) Name() string {
	return "Layer Stack"
}

func (l *layerStack) String() string {
	s := "LayerStack: \n -Activation"
	s += l.layer.activation.String() + "\n"
	s += l.blame.String() + "\n"
	return s
}
