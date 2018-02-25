package main

import (
	"./learnML"
	"./matrix"
	"fmt"
)

func main() {
	n := learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, learnML.Dims{1, 2})
	n.AddLayer(learnML.LayerTanh, learnML.Dims{2})
	n.AddLayer(learnML.LayerLinear, learnML.Dims{2, 1})
	n.InitWeight([]matrix.Vector{{.3, .4, .1, .2}, {}, {.2, .3, .1}})
	x := matrix.Vector{.3}
	y := matrix.Vector{.7}
	n.Activate(&x)
	n.BackProp(y, nil)
	grad := n.CreateGradient()
	n.UpdateGradient(&x, grad)
	rate := 0.1
	n.RefineWeight(grad, rate)
	n.Activate(&x)
	n.BackProp(y, nil)
	n.ScaleGradient(grad, 0.0)
	n.UpdateGradient(&x, grad)
	n.RefineWeight(grad, rate)
	//w, _ := n.Weight()
	//fmt.Printf("weight is %v\n", w)
	n.Activate(&x)
	n.BackProp(y, nil)
	fmt.Printf("n is %v\n", n)
	n.ScaleGradient(grad, 0.0)
	n.UpdateGradient(&x, grad)
	n.RefineWeight(grad, rate)
	w, _ := n.Weight()
	fmt.Printf("weight is %v\n", w)
	fmt.Printf("pred is %v\n", n.Predict(x))
	fmt.Printf("n is %v\n", n)
}
