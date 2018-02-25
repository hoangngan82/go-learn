package learnML

import (
	"../matrix"
)

type layerComposite struct {
	layer
	activation []matrix.Vector
	unit       []Layer
}

func (l *layerComposite) Activate(x *matrix.Vector) *matrix.Vector {
	for i := 0; i < len(l.unit); i++ {
		l.unit[i].Activate(x)
	}
	return &(l.layer.activation)
}

func (l *layerComposite) BackProp(prevBlame *matrix.Vector) {
	var b matrix.Vector
	var start, end int
	for i := 0; i < len(l.unit); i++ {
		end = start + len(l.activation[i])
		b = (*prevBlame)[start:end]
		l.unit[i].BackProp(&b)
		start = end
	}
}

func (l *layerComposite) Name() string {
	return "Layer Comp"
}
