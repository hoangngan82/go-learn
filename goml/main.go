package main

import (
  "fmt"
  "./matrix"
  "./rand"
)

func main() {
  tm := matrix.NewMatrix(1, 3, 2.3, 2.3, 4.5, 8.6)
  fmt.Printf("old matrix is \n%v\n", tm)
  timeZone := "-06:00"
  tm.LoadARFF("/tmp/iris.arff", timeZone)
  fmt.Printf("arff matrix is \n%v\n", tm)
  tm.SwapRows(0, 2)
  fmt.Printf("swapped matrix is \n%v\n", tm)
  tm.SwapCols(0, 5)
  fmt.Printf("swapped matrix is \n%v\n", tm)
  h := matrix.NewMatrix(1, 1)
  //tm = tm.SubMatrix([2]int{0, 0}, [2]int{3, 3})
  //fmt.Printf("sub matrix is \n%v\n", tm)
  h = matrix.Mul(tm, tm, false, true)
  fmt.Printf("mul matrix is \n%v\n", h)
  h.Mul(tm, tm, false, true)
  fmt.Printf("method mul matrix is \n%v\n", h)
  for j:= 0; j < 5; j++ {
    fmt.Println("At j =", j)
  for i:= 0; i < 5; i++ {
    fmt.Printf("name of col %d, val %d is %q\n", i, j, tm.GetValueName(i, j))
    fmt.Printf("name of col %d, val %d is %q\n", i, j, tm.GetAttrType(i))
  }
  }
  tm.SaveARFF("/tmp/myarff")
  r := rand.NewRand(234)
  fmt.Printf("uniform from 0 to 1 is %f\n", r.Uniform())
  nn := uint64(2)
  ii := 0
  run := 0
  for true {
    mm := r.Next(nn)
    if mm % 2 == 1 { run++ }
    if run > 200 { break }
    ii++
  }
  fmt.Println("ii is", ii)
}
