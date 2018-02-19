package learnML

import (
	"../matrix"
	"../rand"
	"gonum.org/v1/gonum/floats"
	//"fmt"
	//"time"
)

// According to new considerations:
// - lines connecting neurons from one set to the next is called a
// layer and represented by layerLinear; each layerLinear has an
// activation and blame vectors as other layer; however, they have an
// additional vector to store weights.
// - each set of neurons is called a layer, which consists of only
// activation function.
type neuralNet struct {
	linearLayer     []layerLinear
	activatingLayer []Layer
}

// InDim return the dimension of the input of a neural network.
func (n *neuralNet) InDim() Dimension {
	d := len(n.linearLayer[0].weight)/len(n.linearLayer[0].activation) - 1
	return Dimension{d}
}

// OutDim return the dimension of the output of a neural network.
func (n *neuralNet) OutDim() Dimension {
	N := len(n.linearLayer)
	return n.linearLayer[N-1].OutDim()
}

func (n *neuralNet) AddLayer(t Layer) {
	linear := layerLinear{}
	linear.init(t.OutDim()[0], n.OutDim()[0])
	n.linearLayer = append(n.linearLayer, linear)
	n.activatingLayer = append(n.activatingLayer, t)
}

// NewNeuralNet creates a neural network. unitsPerLayers determines the
// number of units in each layer. The first layer is the layer after
// the input. The last layer is the output layer. Size of the blame
// vector in each layer is equal to the size of the activation in
// that layer.
func NewNeuralNet(unitsPerLayers []int, layerType LayerType,
	numFeatures, numLabels int) *neuralNet {
	n := &(neuralNet{})
	N := len(unitsPerLayers)

	// initialize linearLayer
	n.linearLayer = make([]layerLinear, N)
	inDim := numFeatures
	for i := 0; i < N; i++ {
		n.linearLayer[i].init(unitsPerLayers[i], inDim)
		inDim = unitsPerLayers[i]
	}

	// initialize Layer
	n.activatingLayer = make([]Layer, N)

	// identify Layer type
	switch layerType {
	case LayerTanh:
		for i := 0; i < N; i++ {
			m := layerTanh{}
			m.init(Dimension{unitsPerLayers[i]})
			n.activatingLayer[i] = &m
		}
	case LayerLeakyRectifier:
		for i := 0; i < N; i++ {
			m := layerLeakyRectifier{}
			m.init(Dimension{unitsPerLayers[i]})
			n.activatingLayer[i] = &m
		}
	default:
		for i := 0; i < N; i++ {
			m := layer{}
			m.init(Dimension{unitsPerLayers[i]})
			n.activatingLayer[i] = &m
		}
	}
	return n
}

func (n *neuralNet) Weight() ([]matrix.Vector, matrix.Vector) {
	totalSize := 0
	for i := 0; i < len(n.linearLayer); i++ {
		totalSize += len(n.linearLayer[i].weight)
	}
	data := matrix.NewVector(totalSize, nil)
	g := make([]matrix.Vector, len(n.linearLayer))
	start := 0
	for i := 0; i < len(n.linearLayer); i++ {
		glen := len(n.linearLayer[i].weight)
		g[i] = data[start : start+glen]
		copy(g[i], n.linearLayer[i].weight)
		start = glen
	}
	return g, data
}

func (n *neuralNet) InitWeight(w []matrix.Vector) {
	if len(w) > 0 {
		for i := 0; i < len(n.linearLayer); i++ {
			copy(n.linearLayer[i].weight, w[i])
		}
		return
	}

	r := rand.NewRand(2162018)
	//r := rand.NewRand(uint64(time.Now().UnixNano()))
	for i := 0; i < len(n.linearLayer); i++ {
		outputCount := len(n.linearLayer[i].activation)
		inputCount := len(n.linearLayer[i].weight)/outputCount - 1
		max := 1.0 / float64(inputCount)
		if max < 0.03 {
			max = 0.03
		}
		for j := 0; j < len(n.linearLayer[i].weight); j++ {
			n.linearLayer[i].weight[j] = max * r.Normal()
		}
	}
}

func (n *neuralNet) Name() string {
	if n == nil {
		return "empty neuralNet"
	}
	s := "\ninput => "
	for i := 0; i < len(n.activatingLayer); i++ {
		s += "Layer Linear -> " + n.activatingLayer[i].Name() + " -> "
	}
	s += "output\n"
	return s
}

// Train trains the model and stores weight in weights from the
// linear layers.
func (n *neuralNet) Train(features, labels *matrix.Matrix) {
	// We will load epochPerPeriod, batchSize, learningRate from a
	// file. If the file does not exist, use default values
	//r := rand.NewRand(uint64(time.Now().UnixNano()))
	r := rand.NewRand(2192018)
	epochPerPeriod := 1
	learningRate := .03
	batchSize := 1
	// load values from file
	rows := labels.Rows()
	if batchSize > rows {
		batchSize = rows
	}
	numBatch := rows / batchSize
	largeBatch := rows % batchSize
	for numBatch < largeBatch {
		batchSize++
		numBatch = rows / batchSize
		largeBatch = rows % batchSize
	}
	gradient := n.CreateGradient()

	// loop through epochs per period
	for j := 0; j < epochPerPeriod; j++ {
		// shuffle data
		for rr := rows; rr > 1; rr-- {
			l := int(r.Next(uint64(rr)))
			features.SwapRows(rr-1, l)
			labels.SwapRows(rr-1, l)
		}

		// now loop through batches
		var start, end int
		batchSize++
		start = 0
		for b := 0; b < largeBatch; b++ {
			end = start + batchSize
			ResetGradient(gradient)
			for s := start; s < end; s++ {
				x := features.Row(s)
				y := labels.Row(s)
				n.Activate(&x)
				n.BackProp(y, nil)
				n.UpdateGradient(&x, gradient)
			}
			//n.RefineWeight(gradient, learningRate/float64(batchSize))
			n.RefineWeight(gradient, learningRate)
			start = end
		}
		batchSize--
		for b := largeBatch; b < numBatch; b++ {
			end = start + batchSize
			ResetGradient(gradient)
			for s := start; s < end; s++ {
				x := features.Row(s)
				y := labels.Row(s)
				n.Activate(&x)
				n.BackProp(y, nil)
				n.UpdateGradient(&x, gradient)
			}
			//fmt.Printf("\nat b = %7d, grad = ", b)
			//for p := 0; p < len(*gradient); p++ {
			//fmt.Printf("%20.15e ", (*gradient)[p].Norm(0))
			//}
			//fmt.Printf("\nweight = ")
			//n.RefineWeight(gradient, learningRate/float64(batchSize))
			n.RefineWeight(gradient, learningRate)
			//weight, _ := n.Weight()
			//for p := 0; p < len(*gradient); p++ {
			//fmt.Printf("%20.15e ", weight[p].Norm(0))
			//}
			start = end
		}
	}
}

func (n *neuralNet) TrainIncremental(feat matrix.Vector, lab matrix.Vector) {
	panic("not implemented")
}

// Activate activates the whole network based on the input in.
func (n *neuralNet) Activate(in *matrix.Vector) *matrix.Vector {
	activation := &matrix.Vector{}
	*activation = make(matrix.Vector, len(*in))
	copy(*activation, *in)

	for i := 0; i < len(n.linearLayer); i++ {
		// apply linear layer
		activation = n.linearLayer[i].Activate(activation)
		// apply activation function
		activation = n.activatingLayer[i].Activate(activation)
	}
	return activation
}

// Predict will call neuralNet.Activate to compute predictions.
func (n *neuralNet) Predict(in matrix.Vector) matrix.Vector {
	return *(n.Activate(&in))
}

// neuralNet.BackProp computes blame for each linear layer. We need
// to feed it with the target vector.
func (n *neuralNet) BackProp(target matrix.Vector, prevBlame *matrix.Vector) {
	N := len(n.activatingLayer)
	output := *(n.activatingLayer[N-1].Activation())
	output = target.Sub(output)
	var m matrix.Vector = matrix.Vector{}
	n.activatingLayer[N-1].BackProp(&m)
	floats.MulTo(n.linearLayer[N-1].blame, output, m)
	//for i := 0; i < len(output); i++ {
	//n.linearLayer[N-1].blame[i] = output[i] * m[i]
	//}

	for i := N - 1; i > 0; i-- {
		n.activatingLayer[i-1].BackProp(&m)
		n.linearLayer[i].BackProp(&(n.linearLayer[i-1].blame))
		floats.Mul(n.linearLayer[i-1].blame, m)
		//for j := 0; j < len(m); j++ {
		//n.linearLayer[i-1].blame[j] *= m[j]
		//}
	}
}

func (n *neuralNet) ResetGradient(gradient *[]matrix.Vector) {
	ResetGradient(gradient)
}

func ResetGradient(g *[]matrix.Vector) {
	gradient := *g
	for i := 0; i < len(gradient); i++ {
		for j := 0; j < len(gradient[i]); j++ {
			gradient[i][j] = 0.0
		}
	}
}

// UpdateGradient requires the neuralNet to be already activated and
// backpropagated.
// tested on 2018-02-09 10:54
func (n *neuralNet) UpdateGradient(x *matrix.Vector,
	g *[]matrix.Vector) {
	gradient := *g
	n.linearLayer[0].UpdateGradient(x, &(gradient[0]))
	for i := 1; i < len(gradient); i++ {
		n.linearLayer[i].UpdateGradient(
			n.activatingLayer[i-1].Activation(),
			&(gradient[i]))
	}
}

func (n *neuralNet) FilterData(featIn *matrix.Matrix, labIn *matrix.Matrix, featOut *matrix.Matrix, labOut *matrix.Matrix) {
	panic("not implemented")
}

// tested on 2018-02-08 11:08
func (n *neuralNet) CentralDifference(in, out *matrix.Vector,
	dt float64, g *[]matrix.Vector) {
	gradient := *g
	var m, p, diff matrix.Vector
	m = *(n.Activate(in))
	diff = (*out).Sub(m)

	for i := 0; i < len(n.linearLayer); i++ {
		for j := 0; j < len(n.linearLayer[i].weight); j++ {
			oldWeight := n.linearLayer[i].weight[j]
			n.linearLayer[i].weight[j] = oldWeight + dt/2.0
			m = *(n.Activate(in))
			p = matrix.NewVector(len(m), nil)
			p.Copy(m)
			n.linearLayer[i].weight[j] = oldWeight - dt/2.0
			m = *(n.Activate(in))
			m = p.Sub(m)
			n.linearLayer[i].weight[j] = oldWeight
			gradient[i][j] = m.Scale(1.0 / dt).Dot(diff)
		}
	}
}

func (n *neuralNet) CreateGradient() *[]matrix.Vector {
	g := make([]matrix.Vector, len(n.linearLayer))
	for i := 0; i < len(n.linearLayer); i++ {
		g[i] = matrix.NewVector(len(n.linearLayer[i].weight), nil)
	}
	return &g
}

func (n *neuralNet) RefineWeight(gradient *[]matrix.Vector, rate float64) {
	g := *gradient
	for i := 0; i < len(n.linearLayer); i++ {
		floats.AddScaled(n.linearLayer[i].weight, rate, g[i])
		//for j := 0; j < len(n.linearLayer[i].weight); j++ {
		//n.linearLayer[i].weight[j] += rate * g[i][j]
		//}
	}
}

// This is to print neural net using fmt.Printf function.
func (n *neuralNet) String() string {
	s := "=======    BEGIN:    =======\n"
	for i := 0; i < len(n.linearLayer); i++ {
		s += "*********************************************************\n"
		s += "linearLayer: \n - Activation:"
		act := n.linearLayer[i].activation
		bla := n.linearLayer[i].blame
		wgh := n.linearLayer[i].weight
		s += act.String()
		s += " - Blame:"
		s += bla.String()
		s += " - Weight:"
		cols := len(act)
		rows := len(wgh) / cols
		s += wgh.ToMatrix(rows, cols).String()
		s += "activatingLayer: " + n.activatingLayer[i].Name() +
			"\n - Activation:"
		s += (*(n.activatingLayer[i].Activation())).String()
	}
	s += "*********************************************************\n"
	s += "=======    END:    =======\n"
	return s
}
