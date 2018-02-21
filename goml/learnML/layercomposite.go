package learnML

import (
	"../matrix"
)

type layerComposite struct {
	layer
	activation []matrix.Vector
	unit       []Layer
}

func NewLayerComposite(first Layer, next ...Layer) *layerComposite {
	var l layerComposite
	l.activation = make([]matrix.Vector, len(next)+1)
	l.unit = make([]Layer, len(next)+1)

	// allocate a continuous storage for activation
	totalSize := first.OutDim()[0]
	for i := 0; i < len(next); i++ {
		totalSize += next[i].OutDim()[0]
	}
	l.layer.activation = matrix.NewVector(totalSize, nil)

	// copy content from inputs
	start := 0
	end := first.OutDim()[0]
	l.activation[0] = l.layer.activation[start:end]
	l.unit[0] = first.Wrap(l.activation[0])
	start = end
	for i := 0; i < len(next); i++ {
		end = start + next[i].OutDim()[0]
		j := i + 1
		l.activation[j] = l.layer.activation[start:end]
		l.unit[j] = next[i].Wrap(l.activation[j])
		start = end
	}
	return &l
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
