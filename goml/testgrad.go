package main

import (
	"./learnML"
	"./matrix"
	"fmt"
)

func main() {
	fmt.Println("Test LayerTanh")
	n := learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, []int{2, 2})
	n.AddLayer(learnML.LayerTanh, []int{2, 2})
	n.AddLayer(learnML.LayerLinear, []int{2, 2})
	n.AddLayer(learnML.LayerTanh, []int{2, 2})
	x := matrix.Vector{.19, .21}
	y := matrix.Vector{0, 0}
	w := []matrix.Vector{
		{.11, .12, .13, .14, .23, .25}, {},
		{.15, .16, .17, .18, .27, .29}, {}}
	n.InitWeight(w)
	n.Activate(&x)
	n.BackProp(y, nil)
	grad := n.CreateGradient()
	n.UpdateGradient(&x, grad)
	fmt.Printf("grad BackProp is %v\n", grad)
	n.ScaleGradient(grad, 0.0)
	var dt float64 = .001
	n.CentralDifference(&x, &y, dt, grad)
	fmt.Printf("grad central is %v\n", grad)
	fmt.Println("Test LayerLeakyRectifier")
	n = learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, []int{2, 2})
	n.AddLayer(learnML.LayerLeakyRectifier, []int{2, 2})
	n.AddLayer(learnML.LayerLinear, []int{2, 2})
	n.AddLayer(learnML.LayerLeakyRectifier, []int{2, 2})
	n.InitWeight(w)
	n.Activate(&x)
	n.BackProp(y, nil)
	n.ScaleGradient(grad, 0.0)
	n.UpdateGradient(&x, grad)
	fmt.Printf("grad BackProp is %v\n", grad)
	n.ScaleGradient(grad, 0.0)
	n.CentralDifference(&x, &y, dt, grad)
	fmt.Printf("grad central is %v\n", grad)

	n = learnML.NewNeuralNet()
	n.AddLayer(learnML.LayerLinear, []int{1, 3}, []int{1, 3})
	n.AddLayer(learnML.LayerSinusoidal, []int{2, 1})
	n.AddLayer(learnML.LayerLinear, []int{3, 1}, []int{0, 1}, []int{1, 7})
	grad = n.CreateGradient()
	fmt.Println(n.Structure())
	w = []matrix.Vector{
		{-.11, .12, .13, .14, .15, .16}, {},
		{.15, .16, .17, -.1156584681688}}
	x = matrix.Vector{.1}
	y = matrix.Vector{.3}
	n.InitWeight(w)
	n.Activate(&x)
	n.BackProp(y, nil)
	n.ScaleGradient(grad, 0.0)
	fmt.Printf("n is %v\n", n)
	n.UpdateGradient(&x, grad)
	fmt.Printf("grad BackProp is %v\n", grad)
	//n.ScaleGradient(grad, 0.0)
	//n.CentralDifference(&x, &y, dt, grad)
	//fmt.Printf("grad central is %v\n", grad)
	//x = matrix.Vector{.85637, .35165}
	//fmt.Printf("%v\n", n.Activate(&x))
	//n.ScaleGradient(grad, 0.0)
	//n.Activate(&x)
	//n.BackProp(y, nil)
	//n.UpdateGradient(&x, grad)
	//fmt.Printf("grad BackProp is %v\n", grad)
	//n.ScaleGradient(grad2, 0.0)
	//n.CentralDifference(&x, &y, dt, grad2)
	//fmt.Printf("grad central is %v\n", grad)
	//fmt.Println("difference between CentralDifference and BackProp")
	//for i := 0; i < len(*grad); i++ {
	//fmt.Printf("Layer %d: %e\n", i, ((*grad)[i].Sub((*grad2)[i]).Norm(2)))
	//}
}
