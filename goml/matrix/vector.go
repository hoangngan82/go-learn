// Package vector implement a mathematical vector together with some
// metadata. Each element is represented as a float64 (double).
// Matrices are stored in column-major order and all columns have a
// name.
package matrix

import (
  "fmt"
  "bytes"
)

type Vector struct {
  data  []float64
  size  int
}

// ToMatrix wraps a 1-by-n matrix around the underlying data of a
// vector.
func (v *Vector) ToMatrix() *Matrix {
  var m Matrix
  m.rows = 1
  m.cols = v.size
  cols := m.cols

  // metadata
  m.relation = "default"
  m.attrName = make ([]string, cols, cols + EXTRA_NUM_CELL)
  m.str_to_enum = make ([] map[string]int, cols, cols + EXTRA_NUM_CELL)
  m.enum_to_str = make ([] map[int]string, cols, cols + EXTRA_NUM_CELL)
  for i:= 0; i < cols; i++ {
    m.attrName[i] = fmt.Sprintf("col_%d",i)
    m.enum_to_str[i] = make (map[int]string)
    m.enum_to_str[i][ATTR_NAME] = "real"
  }

  // actual data
  m.data = make ([][]float64, 1)
  m.data[0] = v.data
  return &m
}

// NewVector creates a vector of length size and set all element
// to the value vals (default = 0).
// NewVector(5, 4.3, 3.2, 2.1) gives [4.3, 3.2, 2.1, 4.3, 3.2]
func NewVector(size int, val ...float64) *Vector {
  var v Vector
  v.size = size
  v.data = make([]float64, size + EXTRA_NUM_CELL)
  N := len(val)
  if N == 0 {
    return &v
  }

  for i := 0; i < size; i++ {
    v.data[i] = val[i%N]
  }
  return &v
}

// SetElem sets the element at index i to val.
func (v *Vector) SetElem(i int, val float64) {
  Require(i >= 0 && i < v.size,
    "SetElem: index out of bound: i = %d\n", i)
  v.data[i] = val
}

// SetElems sets the elements at indices to val's.
func (v *Vector) SetElems(indices []int, vals []float64) {
  Require(len(indices) != 0 && len(vals) != 0,
    "SetElems: empty index: len(indices) = %d, len(vals) = %d\n",
    len(indices), len(vals))
  Require(len(indices) == len(vals),
    "SetElems: dimension mismatched: len(indices) = %d, " +
    "len(vals) = %d\n", len(indices), len(vals))
  for i, val := range vals {
    Require(indices[i] < v.size && indices[i] >= 0,
      "SetElems: index out of bound: indices[%d] = %d\n",
      i, indices[i])
    v.data[indices[i]] = val
  }
}

// GetElems returns the elements at indices.
func (v *Vector) GetElems(indices []int) *Vector {
  Require(len(indices) != 0,
    "SetElems: empty index: len(indices) = %d\n",
    len(indices))
  for i := 0; i < len(indices); i++ {
    Require(indices[i] < v.size && indices[i] >= 0,
      "SetElems: index out of bound: indices[%d] = %d\n",
      i, indices[i])
  }
  var e Vector
  e.size = len(indices)
  e.data = make([]float64, e.size)
  for i := 0; i < e.size; i++ {
    e.data[i] = v.data[indices[i]]
  }
  return &e
}

// GetElem returns the element at index i in the Vector.
func (v *Vector) GetElem(i int) float64 {
  Require(i >= 0 && i < v.size,
    "SetElem: index out of bound: i = %d\n", i)
  return v.data[i]
}

// String converts a Vector into a string so that it can be printed
// using fmt.Printf("%v\n", vector).
func (v *Vector) String() string {
  var buf bytes.Buffer
  buf.Grow(v.size*22 + 3 + 2)
  fmt.Fprintf(&buf, "\n[\n")
  for i:= 0; i < v.size; i++ {
    fmt.Fprintf(&buf, " %+20.12e\n", v.data[i])
  }
  fmt.Fprintf(&buf, "]\n")
  return buf.String()
}

// Size return the size of a vector
func (v *Vector) Size() int {
  return v.size
}

// add returns a + sign*b
func (a *Vector) add(b *Vector, sign int) *Vector {
  Require(a.size == b.size,
    "Add: dimension mismatch: a.size == b.size\n")
  s := NewVector(a.size)
  if sign < 0 {
    for i:= 0; i < s.size; i++ {
      s.data[i] = a.data[i] - b.data[i]
    }
  } else {
    for i:= 0; i < s.size; i++ {
      s.data[i] = a.data[i] + b.data[i]
    }
  }
  return s
}

// Sub returns the difference a - b.
func (a *Vector) Sub(b *Vector) *Vector {
  return a.add(b, -1)
}

// Add returns the sum a + b.
func (a *Vector) Add(b *Vector) *Vector {
  return a.add(b, 1)
}

// Dot returns the dot product of two vectors.
func (a *Vector) Dot(b *Vector) float64 {
  Require(a.size == b.size,
    "Add: dimension mismatch: a.size == b.size\n")
  d := float64(0)
  for i:= 0; i < a.size; i++ {
    d += a.data[i]*b.data[i]
  }
  return d
}

// OuterProd returns the matrix a^t*b.
func (a *Vector) OuterProd(b *Vector) *Matrix {
  o := NewMatrix(a.size, b.size)
  for i:= 0; i < o.rows; i++ {
    for j:= 0; j < o.cols; j++ {
      o.data[i][j] = a.data[i]*b.data[j]
    }
  }
  return o
}

