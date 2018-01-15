package main

import (
  "fmt"
  "./matrix"
)

func main() {
  // create matrix [[1 2 3] [-1 0 1]]
  start := matrix.NewMatrix(2, 3, 1, 2, 3, -1, 0, 1)
  fmt.Printf("matrix [[1 2 3] [-1 0 1]] is %v", start)
  start.ChangeAttrName([]string{"", "experience", ""})
  fmt.Printf("matrix [[1 2 3] [-1 0 1]] is %v", start)

  fmt.Printf("S*S^t is %v\n", matrix.Mul(start, start, false, true))
  fmt.Printf("S^t*S is %v\n", matrix.Mul(start, start, true, false))

  v := matrix.NewVector(2, 1, -2)
  w := matrix.NewVector(3, 1, -1, 2)
  fmt.Printf("A*v + v is %v\n", matrix.Axpb(start, w, v))
  fmt.Printf("A*v + v is %v\n", matrix.Axmb(start, w, v))

  start.SaveARFF("/tmp/test.arff")
  fmt.Printf("vw is %v\n", v.OuterProd(w))
  fmt.Printf("wv is %v\n", w.OuterProd(v))
}
