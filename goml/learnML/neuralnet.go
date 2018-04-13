package learnML

import (
	"../matrix"
	"../rand"
	//"fmt"
	//"gonum.org/v1/gonum/floats"
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
	layers []Layer
}

// OutDim return the dimension of the output of a neural network.
func (n *neuralNet) OutDim() Dims {
	N := len(n.layers)
	return n.layers[N-1].OutDim()
}

func (n *neuralNet) AddLayer(t LayerType, dim Dims, dims ...Dims) {
	l := NewLayer(t, dim, dims...)
	n.layers = append(n.layers, l)
}

// NewNeuralNet creates a neural network. unitsPerLayers determines the
// number of units in each layer. The first layer is the layer after
// the input. The last layer is the output layer. Size of the blame
// vector in each layer is equal to the size of the activation in
// that layer.
func NewNeuralNet() *neuralNet {
	n := neuralNet{}
	n.layers = make([]Layer, 0, 4)
	return &n
}

func (n *neuralNet) Weight() ([]matrix.Vector, matrix.Vector) {
	totalSize := 0
	for i := 0; i < len(n.layers); i++ {
		totalSize += len(*(n.layers[i].Weight()))
	}
	data := matrix.NewVector(totalSize, nil)
	g := make([]matrix.Vector, len(n.layers))
	start := 0
	for i := 0; i < len(n.layers); i++ {
		glen := len(*(n.layers[i].Weight()))
		g[i] = data[start : start+glen]
		copy(g[i], *(n.layers[i].Weight()))
		start = glen
	}
	return g, data
}

func (n *neuralNet) CopyWeight() []matrix.Vector {
	g := make([]matrix.Vector, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		g[i] = matrix.NewVector(len(*(n.layers[i].Weight())), nil)
		copy(g[i], *(n.layers[i].Weight()))
	}
	return g
}

//func (n *neuralNet) CopyWeight() []matrix.Vector {
//g := make([]matrix.Vector, len(n.layers))
//for i := 0; i < len(n.layers); i++ {
//g[i] = *(n.layers[i].Weight())
//}
//return g
//}

// Initializing weights with a constant will keep the weights vector
// a constant vector (with different values of the constant, maybe).
func (n *neuralNet) InitWeight(w []matrix.Vector) {
	if len(w) > 0 {
		for i := 0; i < len(n.layers); i++ {
			copy(*(n.layers[i].Weight()), w[i])
		}
		return
	}

	r := rand.NewRand(2162018)
	//r := rand.NewRand(uint64(time.Now().UnixNano()))
	for i := 0; i < len(n.layers); i++ {
		outputCount := len(*(n.layers[i].Activation()))
		if outputCount == 0 {
			continue
		}
		inputCount := len(*(n.layers[i].Weight()))/outputCount - 1
		max := 1.0
		if inputCount != 0 {
			max /= float64(inputCount)
		}
		if max < 0.03 {
			max = 0.03
		}
		for j := 0; j < len(*(n.layers[i].Weight())); j++ {
			(*(n.layers[i].Weight()))[j] = max * r.Normal()
		}
	}
}

func (n *neuralNet) Structure() string {
	if n == nil {
		return "empty neuralNet"
	}
	s := "\ninput => "
	for i := 0; i < len(n.layers); i++ {
		s += n.layers[i].Name() + " -> "
	}
	s += "output\n"
	return s
}

func (n *neuralNet) Name() string {
	return n.Structure()
}

// Train trains the model and stores weight in weights from the
// linear layers. Train only runs for one full epoch, if you want to
// train for N epochs, you have to call Train N times.
// There only 4 parameters can be passed through the map params, namely,
// seed, learningRate, batchSize, and momentum.
func (n *neuralNet) Train(features, labels *matrix.Matrix,
	params map[string]float64) {
	matrix.Require(features.Rows() == labels.Rows(),
		"neuralNet.Train: Expect %s but get %d = %d\n",
		"features.Rows() == labels.Rows()", features.Rows(), labels.Rows())

	seed := uint64(params["seed"])
	learningRate := 0.03
	batchSize := 1
	momentum := 0.0

	rows := labels.Rows()

	if params["learningRate"] > 0.0 {
		learningRate = params["learningRate"]
	}

	if params["momentum"] > 0.0 {
		momentum = params["momentum"]
	}

	var temp int = 0
	temp = int(params["batchSize"])
	if temp != 0 {
		batchSize = temp
	}
	if batchSize > rows {
		batchSize = rows
	}

	// compute batch size
	numBatch := rows / batchSize
	largeBatch := rows % batchSize
	for numBatch < largeBatch {
		batchSize++
		numBatch = rows / batchSize
		largeBatch = rows % batchSize
	}
	gradient := n.CreateGradient()

	P := make([]int, features.Rows())
	for i := 0; i < len(P); i++ {
		P[i] = i
	}
	// shuffle data
	random := rand.NewRand(seed)
	for r := rows; r > 1; r-- {
		l := int(random.Next(uint64(r)))
		P[r-1], P[l] = P[l], P[r-1]
	}

	// now loop through batches
	var start, end int
	batchSize++
	start = 0
	for b := 0; b < largeBatch; b++ {
		end = start + batchSize
		ScaleGradient(gradient, momentum)
		for s := start; s < end; s++ {
			x := features.Row(P[s])
			y := labels.Row(P[s])
			n.Activate(&x)
			n.BackProp(y, nil)
			n.UpdateGradient(&x, gradient)
		}
		n.RefineWeight(gradient, learningRate/float64(batchSize))
		//n.RefineWeight(gradient, learningRate)
		start = end
	}
	batchSize--
	for b := largeBatch; b < numBatch; b++ {
		end = start + batchSize
		ScaleGradient(gradient, momentum)
		for s := start; s < end; s++ {
			x := features.Row(P[s])
			y := labels.Row(P[s])
			n.Activate(&x)
			n.BackProp(y, nil)
			n.UpdateGradient(&x, gradient)
		}
		n.RefineWeight(gradient, learningRate/float64(batchSize))
		start = end
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

	for i := 0; i < len(n.layers); i++ {
		activation = n.layers[i].Activate(activation)
	}
	return activation
}

// Predict will call neuralNet.Activate to compute predictions.
func (n *neuralNet) Predict(in matrix.Vector) matrix.Vector {
	return *(n.Activate(&in))
}

// neuralNet.BackProp computes blame for each linear layer. We need
// to feed it with the target vector.
// Note that this BackProp function omit the constant 2 in front of
// the actual derivative.
func (n *neuralNet) BackProp(target matrix.Vector, prevBlame *matrix.Vector) {
	N := len(n.layers)
	output := *(n.layers[N-1].Activation())
	output = target.Sub(output)

	copy(*(n.layers[N-1].Blame()), output)

	for i := N - 1; i > 0; i-- {
		n.layers[i].BackProp(n.layers[i-1].Blame())
	}
}

func (n *neuralNet) ScaleGradient(gradient *[]matrix.Vector, c float64) {
	ScaleGradient(gradient, c)
}

func ScaleGradient(g *[]matrix.Vector, c float64) {
	gradient := *g
	for i := 0; i < len(gradient); i++ {
		v := gradient[i]
		for j := 0; j < len(v); j++ {
			v[j] *= c
		}
		//floats.Scale(c, gradient[i])
	}
}

// UpdateGradient requires the neuralNet to be already activated and
// backpropagated.
// tested on 2018-02-09 10:54
func (n *neuralNet) UpdateGradient(x *matrix.Vector,
	g *[]matrix.Vector) {
	gradient := *g
	n.layers[0].UpdateGradient(x, &(gradient[0]))
	for i := 1; i < len(gradient); i++ {
		n.layers[i].UpdateGradient(
			n.layers[i-1].Activation(),
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

	for i := 0; i < len(n.layers); i++ {
		w := *(n.layers[i].Weight())
		for j := 0; j < len(w); j++ {
			oldWeight := w[j]
			w[j] = oldWeight + dt/2.0
			m = *(n.Activate(in))
			p = matrix.NewVector(len(m), nil)
			p.Copy(m)
			w[j] = oldWeight - dt/2.0
			m = *(n.Activate(in))
			m = p.Sub(m)
			w[j] = oldWeight
			gradient[i][j] = m.Scale(1.0 / dt).Dot(diff)
		}
	}
}

func (n *neuralNet) CreateGradient() *[]matrix.Vector {
	g := make([]matrix.Vector, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		g[i] = matrix.NewVector(len(*(n.layers[i].Weight())), nil)
	}
	return &g
}

func (n *neuralNet) RefineWeight(gradient *[]matrix.Vector, rate float64) {
	g := *gradient
	for i := 0; i < len(n.layers); i++ {
		if len(g[i]) == 0 {
			continue
		}
		w := *(n.layers[i].Weight())
		v := g[i]
		for j := 0; j < len(v); j++ {
			w[j] += rate * v[j]
		}
		//floats.AddScaled(*(n.layers[i].Weight()), rate, g[i])
	}
}

// This is to print neural net using fmt.Printf function.
func (n *neuralNet) String() string {
	s := "=======    BEGIN: neuralNet    =======\n"
	for i := 0; i < len(n.layers); i++ {
		s += "*********************************************************\n"
		s += n.layers[i].Name() + "\n - Activation:"
		act := *(n.layers[i].Activation())
		bla := *(n.layers[i].Blame())
		wgh := *(n.layers[i].Weight())
		s += act.String()
		s += " - Blame:"
		s += bla.String()
		s += " - Weight:"
		if len(wgh) == 0 {
			s += " empty []\n"
		} else {
			//cols := len(act)
			//rows := len(wgh) / cols
			//s += wgh.ToMatrix(rows, cols).String()
			s += wgh.String()
		}
	}
	s += "*********************************************************\n"
	s += "=======    END: neuralNet    =======\n"
	return s
}
