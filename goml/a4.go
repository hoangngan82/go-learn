package main

import (
	"./learnML"
	"./matrix"
	"fmt"
	"time"
)

func main() {
	nn := learnML.NewNeuralNet()
	nn.AddLayer(learnML.LayerConv, []int{8, 8}, []int{5, 5, 4}, []int{4, 4, 4})
	nn.AddLayer(learnML.LayerLeakyRectifier, []int{8 * 8 * 4})
	nn.AddLayer(learnML.LayerMaxPooling2D, []int{8, 8, 4})
	nn.AddLayer(learnML.LayerConv, []int{4, 4, 4}, []int{3, 3, 4, 6}, []int{4, 4, 1, 6})
	nn.AddLayer(learnML.LayerLeakyRectifier, []int{4 * 4 * 6})
	nn.AddLayer(learnML.LayerMaxPooling2D, []int{4, 4, 1 * 6})
	nn.AddLayer(learnML.LayerLinear, []int{2 * 2 * 6, 3})
	//nn.AddLayer(learnML.LayerLinear, []int{1 * 1 * 2, 1})
	//w := []matrix.Vector{{.1, .2, .3, .4, .5, .6, .7, .8}, {}, {.05, .1, .15, .2, .25, .3, .35, .4, .45}}
	//w := []matrix.Vector{{.1, .2, .3, .4, .5, .6, .7, .8}, {}, {.05, .1, .15}}
	nn.InitWeight(nil)
	//fmt.Printf("initially, nn is %v\n", nn)
	x := matrix.NewVector(8*8, nil)
	// Error with dt = .01 but not with dt = .005
	//t for x is 1519927477614994046
	//t for y is 1519927477615055489
	//0: 2.5717595e-11
	//3: 2.4057914e-01
	//6: 2.8829784e-12
	// Error with dt = .005 but not with dt = .001
	//t for x is 1519930923103990539
	//t for y is 1519930923104042344
	//0: 3.6108165e-03
	//3: 4.5812087e-11
	//6: 2.3706004e-11

	t := time.Now().UnixNano()
	//t = 1519927477614994046
	//t = 1519930923103990539
	fmt.Printf("Random seed for x is %d\n", t)
	x.Random(t)
	//x = matrix.Vector{.11, .12, .13, .14}
	//x = matrix.Vector{.11, .12, .13, .14, .15, .16, .17, .18, .19}
	//y := matrix.Vector{.1, .2, .3}
	y := matrix.NewVector(3, nil)
	t = time.Now().UnixNano() + 1
	//t = 1519927477615055489
	//t = 1519930923104042344
	fmt.Printf("Random seed for y is %d\n", t)
	y.Random(t)
	//y := matrix.Vector{1.20375}
	nn.Activate(&x)
	//fmt.Printf("activated nn is %v\n", nn)
	nn.BackProp(y, nil)
	//fmt.Printf("BackProp nn is %v\n", nn)
	grad := nn.CreateGradient()
	grad2 := nn.CreateGradient()
	nn.UpdateGradient(&x, grad)
	//fmt.Printf("grad is %v\n", grad)
	//x = matrix.Vector{.11, .12, .13, .14, .15, .16, .17, .18, .19}
	//y = matrix.Vector{1.20375}
	//fmt.Printf("grad2 is %v\n", grad2)
	nn.CentralDifference(&x, &y, .001, grad2)
	N := len(*grad)
	fmt.Println("Difference between BackProp and CentralDifference is")
	for i := 0; i < N; i++ {
		if len((*grad)[i]) == 0 {
			continue
		}
		fmt.Printf("At layer %2d: %.7e\n", i, (*grad)[i].Sub((*grad2)[i]).Norm(1))
	}
	//if i == 3 {
	//for j := 0; j < len((*grad)[i]); j++ {
	//fmt.Printf("%+20.15e  %+20.15e  %+20.15e\n", (*grad)[i][j], (*grad2)[i][j], (*grad)[i].Sub((*grad2)[i])[j])
	//}
	////fmt.Printf("grad is %v\ngrad2 is %v\n", (*grad)[i], (*grad)[i].Sub((*grad2)[i]))
	//}
	//fmt.Printf("grad is %v\n", grad2)
}
