package learnML

import (
	"../matrix"
	"../rand"
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
	d := len(n.linearLayer[0].weight)/len(n.linearLayer[0].layer.activation) - 1
	return Dimension{d}
}

// OutDim return the dimension of the output of a neural network.
func (n *neuralNet) OutDim() Dimension {
	N := len(n.linearLayer)
	return n.linearLayer[N-1].OutDim()
}

func (n *neuralNet) AddLayer(t Layer) {
	linear := layerLinear{}
	linear.Init(t.OutDim(), n.OutDim())
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
		n.linearLayer[i].Init(Dimension{unitsPerLayers[i]},
			Dimension{inDim})
		inDim = unitsPerLayers[i]
	}

	// initialize Layer
	n.activatingLayer = make([]Layer, N)

	// identify Layer type
	switch layerType {
	case LayerTanh:
		for i := 0; i < N; i++ {
			m := layerTanh{}
			m.Init(Dimension{unitsPerLayers[i]})
			n.activatingLayer[i] = &m
		}
	case LayerLeakyRectifier:
		for i := 0; i < N; i++ {
			m := layerLeakyRectifier{}
			m.Init(Dimension{unitsPerLayers[i]})
			n.activatingLayer[i] = &m
		}
	default:
		for i := 0; i < N; i++ {
			m := layer{}
			m.Init(Dimension{unitsPerLayers[i]})
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

	r := rand.NewRand(262018)
	for i := 0; i < len(n.linearLayer); i++ {
		max := 1.0 / float64(len(n.linearLayer[i].weight))
		if max < 0.03 {
			max = 0.03
		}
		for j := 0; j < len(n.linearLayer[i].weight); j++ {
			n.linearLayer[i].weight[j] = 0.03 * r.Normal()
			// for testing: tested on 2018-02-07 21:58
			//n.linearLayer[i].weight[j] = float64(i + j) // 100.0
		}
	}
}

func (n *neuralNet) Name() string {
	if n == nil {
		return "empty neuralNet"
	}
	var layerType interface{}
	layerType = n.linearLayer[0]
	switch layerType.(type) {
	case *layerTanh:
		return "neuralNet: layerTanh"
	default:
		return "neuralNet: layerLinear"
	}
}

// Train trains the model and stores weight in weight[0].
func (n *neuralNet) Train(features, labels *matrix.Matrix) {
	var layerType interface{}
	layerType = n.linearLayer[0]
	switch layerType.(type) {
	case *layerTanh:
		panic("Currently, only layerLinear is supported")
	default:
		n.linearLayer[0].weight = matrix.OLS(features, labels)
	}
}

func (n *neuralNet) TrainIncremental(feat matrix.Vector, lab matrix.Vector) {
	panic("not implemented")
}

// tested on 2018-02-07 21:59
//func main() {
//var n learnML.neuralNet
//n.Init([]int{2, 3}, learnML.layerTanh{}, 2, 3)
//n.InitWeight()
//x := matrix.Vector{1, 1}
//fmt.Printf("activation is %v\n", n.Activate(&x))
//fmt.Printf("n is %v\n", n)
//x = matrix.Vector{1, -1}
//fmt.Printf("activation is %v\n", n.Activate(&x))
//fmt.Printf("n is %v\n", n)
//}
// Activate activates the whole network based on the input in.
func (n *neuralNet) Activate(in *matrix.Vector) *matrix.Vector {
	activation := in
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
	output = target.Sub(output).Scale(2.0)
	var m matrix.Vector = matrix.Vector{}
	n.activatingLayer[N-1].BackProp(&m)
	for i := 0; i < len(output); i++ {
		n.linearLayer[N-1].layer.blame[i] = output[i] * m[i]
	}

	for i := N - 1; i > 0; i-- {
		n.activatingLayer[i-1].BackProp(&m)
		n.linearLayer[i].BackProp(n.linearLayer[i-1].Blame())
		for j := 0; j < len(m); j++ {
			n.linearLayer[i-1].layer.blame[j] *= m[j]
		}
	}
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
			gradient[i][j] = 2.0 * m.Scale(1.0/dt).Dot(diff)
		}
	}
}

func (n *neuralNet) CreateGradient() *[]matrix.Vector {
	var g *[]matrix.Vector = &([]matrix.Vector{})
	*g = make([]matrix.Vector, len(n.linearLayer))
	for i := 0; i < len(n.linearLayer); i++ {
		(*g)[i] = matrix.NewVector(len(n.linearLayer[i].weight), nil)
	}
	return g
}

func (n *neuralNet) RefineWeight(gradient *[]matrix.Vector, rate float64) {
	g := *gradient
	for i := 0; i < len(n.linearLayer); i++ {
		for j := 0; j < len(n.linearLayer[i].weight); j++ {
			n.linearLayer[i].weight[j] += rate * g[i][j]
		}
	}
}

// This is to print neural net using fmt.Printf function.
func (n *neuralNet) String() string {
	s := "=======    BEGIN: " + n.Name() + " =======\n"
	for i := 0; i < len(n.linearLayer); i++ {
		s += "*********************************************************\n"
		s += "linearLayer: \n - Activation:"
		act := n.linearLayer[i].layer.activation
		bla := n.linearLayer[i].layer.blame
		wgh := n.linearLayer[i].weight
		s += act.String()
		s += " - Blame:"
		s += bla.String()
		s += " - Weight:"
		cols := len(act)
		rows := len(wgh) / cols
		s += wgh.ToMatrix(rows, cols).String()
		s += "activatingLayer:\n - Activation:"
		s += (*(n.activatingLayer[i].Activation())).String()
		s += " - Blame:"
		s += (*(n.activatingLayer[i].Blame())).String()
	}
	s += "*********************************************************\n"
	s += "=======    END: " + n.Name() + " =======\n"
	return s
}
