package learnML

import (
	"../matrix"
)

type layerOmni struct {
	layer
	activation []matrix.Vector
	unit       []Layer
}

func NewLayerOmni(first Layer, next ...Layer) *layerOmni {
	var l layerOmni

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
	l.unit[0] = first.Copy(l.activation[0])
	start = end
	for i := 0; i < len(next); i++ {
		end = next[i].OutDim()[0]
		j := i + 1
		l.activation[j] = l.layer.activation[start:end]
		l.unit[j] = next[i].Copy(l.activation[j])
		start = end
	}
	return &l
}

func (l *layerOmni) Activate(x *matrix.Vector) *matrix.Vector {
	for i := 0; i < len(l.unit); i++ {
		l.unit[i].Activate(x)
	}
	return &(l.layer.activation)
}

func (l *layerOmni) BackProp(prevBlame *matrix.Vector) {
	var b matrix.Vector
	var start, end int
	for i := 0; i < len(l.unit); i++ {
		end = start + len(l.activation[i])
		b = (*prevBlame)[start:end]
		l.unit[i].BackProp(&b)
		start = end
	}
}

func (l *layerOmni) Name() string {
	return "Layer Omni"
}
