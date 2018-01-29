package main

import (
	"./learnML"
	"./matrix"
	"fmt"
)

func main() {
	// create matrix [[1 2 3] [-1 0 1]]
	start := matrix.NewMatrix(2, 3, matrix.Vector{1, 2, 3, -1, 0, 1})
	fmt.Printf("matrix [[1 2 3] [-1 0 1]] is %v", start)
	start.ChangeAttrName([]string{"", "experience", ""})
	fmt.Printf("matrix [[1 2 3] [-1 0 1]] is %v", start)

	fmt.Printf("S*S^t is %v\n", matrix.Mul(start, start, false, true))
	fmt.Printf("S^t*S is %v\n", matrix.Mul(start, start, true, false))

	v := matrix.Vector{1, -2}
	w := matrix.Vector{1, -1, 2}
	fmt.Printf("A*v + v is %v\n", matrix.Axpb(start, w, v))
	fmt.Printf("A*v + v is %v\n", matrix.Axmb(start, w, v))

	start.SaveARFF("/tmp/test.arff")
	fmt.Printf("vw is %v\n", v.OuterProd(w))
	fmt.Printf("wv is %v\n", w.OuterProd(v))
	timeZone := "-06:00"
	start.LoadARFF("/tmp/iris.arff", timeZone).Random()
	//start.Random()
	fmt.Printf("seed 1982 is %v\n", start)
	start.Random(1984)
	fmt.Printf("seed 1984 is %v\n", start)
	start.AddCols(1).FillCol(-1, 2.3)
	fmt.Printf("AddCols is %v\n", start)
	start.AddRows(1).FillRow(-1, -23)
	fmt.Printf("AddRows is %v\n", start)
	P := []int{2, 3, 0, 1}
	start.PermuteRows(P)
	fmt.Printf("PermuteRows is %v\n", start)
	P = []int{2, 3, 0, 1, 5, 4}
	start.PermuteCols(P)
	fmt.Printf("PermuteRows is %v\n", start)
	var l learnML.LayerLinear
	var n learnML.NeuralNet
	n.Initialize(1, l)
}
