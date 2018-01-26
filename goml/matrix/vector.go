// Package vector implement a mathematical vector together with some
// metadata. Each element is represented as a float64 (double).
// Matrices are stored in column-major order and all columns have a
// name.
package matrix

import (
	"bytes"
	"fmt"
	"math"
)

type Vector []float64

// ToMatrix wraps a 1-by-n matrix around the underlying data of a
// vector.
func (v Vector) ToMatrix() *Matrix {
	return NewMatrix(1, len(v), v)
}

// NewVector creates a vector of length size and set all element
// to the value vals (default = 0).
// NewVector(5, 4.3, 3.2, 2.1) gives [4.3, 3.2, 2.1, 4.3, 3.2]
func NewVector(size int, val ...float64) Vector {
	v := make([]float64, size, size+EXTRA_NUM_CELL)
	N := len(val)
	if N == 0 {
		return v
	}

	for i := 0; i < size; i++ {
		v[i] = val[i%N]
	}
	return v
}

// SetElem sets the element at index i to val.
func (v Vector) SetElem(i int, val float64) {
	Require(i >= 0 && i < len(v),
		"SetElem: index out of bound: i = %d\n", i)
	v[i] = val
}

// SetElems sets the elements at indices to val's.
func (v Vector) SetElems(indices []int, vals []float64) {
	Require(len(indices) != 0 && len(vals) != 0,
		"SetElems: empty index: len(indices) = %d, len(vals) = %d\n",
		len(indices), len(vals))
	Require(len(indices) == len(vals),
		"SetElems: dimension mismatched: len(indices) = %d, "+
			"len(vals) = %d\n", len(indices), len(vals))
	for i, val := range vals {
		v[indices[i]] = val
	}
}

// GetElems returns the elements at indices.
func (v Vector) GetElems(indices []int) Vector {
	Require(len(indices) != 0,
		"SetElems: empty index: len(indices) = %d\n",
		len(indices))
	for i := 0; i < len(indices); i++ {
		Require(indices[i] < len(v) && indices[i] >= 0,
			"SetElems: index out of bound: indices[%d] = %d\n",
			i, indices[i])
	}
	e := NewVector(len(indices))
	for i := 0; i < len(e); i++ {
		e[i] = v[indices[i]]
	}
	return e
}

// GetElem returns the element at index i in the Vector.
func (v Vector) GetElem(i int) float64 {
	Require(i >= 0 && i < len(v),
		"SetElem: index out of bound: i = %d\n", i)
	return v[i]
}

// String converts a Vector into a string so that it can be printed
// using fmt.Printf("%v\n", vector).
func (v Vector) String() string {
	var buf bytes.Buffer
	buf.Grow(len(v)*22 + 3 + 2)
	fmt.Fprintf(&buf, "\n[\n")
	for i := 0; i < len(v); i++ {
		fmt.Fprintf(&buf, " %+20.12e\n", v[i])
	}
	fmt.Fprintf(&buf, "]\n")
	return buf.String()
}

// Size return the size of a vector
func (v Vector) Size() int {
	return len(v)
}

// add returns a + sign*b and stores the result in vector a.
func (a Vector) add(b Vector, sign int) {
	Require(len(a) == len(b),
		"Add: dimension mismatch: len(a) == len(b)\n")
	if sign < 0 {
		for i := 0; i < len(a); i++ {
			a[i] -= b[i]
		}
	} else {
		for i := 0; i < len(a); i++ {
			a[i] += b[i]
		}
	}
}

// Sub returns the difference a - b.
func (a Vector) Sub(b Vector) Vector {
	s := NewVector(len(a))
	copy(s, a)
	s.add(b, -1)
	return s
}

// Add returns the sum a + b.
func (a Vector) Add(b Vector) Vector {
	s := NewVector(len(a))
	copy(s, a)
	s.add(b, 1)
	return s
}

// Dot returns the dot product of two vectors.
func (a Vector) Dot(b Vector) float64 {
	Require(len(a) == len(b),
		"Add: dimension mismatch: len(a) == len(b)\n")
	d := float64(0)
	for i := 0; i < len(a); i++ {
		d += a[i] * b[i]
	}
	return d
}

// OuterProd returns the matrix a^t*b.
func (a Vector) OuterProd(b Vector) *Matrix {
	o := NewMatrix(len(a), len(b), nil)
	for i := 0; i < o.rows; i++ {
		for j := 0; j < o.cols; j++ {
			o.matrix[i][j] = a[i] * b[j]
		}
	}
	return o
}

// Norm returns the p-norm of a vector where 0-norm is the infinity
// norm. |v|_p = (sum|v_i|^p)^(1/p)
// Only p = 0, 1, 2 are supported.
// Any value of p other than 0 and 1 will result in 2-norm.
func (v Vector) Norm(p int) float64 {
	var n float64 = 0
	switch p {
	case 0: // infinity norm
		for i := 0; i < len(v); i++ {
			if v[i] > n {
				n = v[i]
			} else {
				if v[i] < -n {
					n = -v[i]
				}
			}
		}
		return n
	case 1:
		for i := 0; i < len(v); i++ {
			if v[i] >= 0 {
				n += v[i]
			} else {
				n -= v[i]
			}
		}
		return n
	case 2:
		for i := 0; i < len(v); i++ {
			n += v[i] * v[i]
		}
		return math.Sqrt(n)
	default:
		p2 := float64(p)
		for i := 0; i < len(v); i++ {
			n += math.Pow(v[i], p2)
		}
		return math.Pow(n, 1.0/p2)
	}
}

// Permute rearrange the elements of the vector according to a
// permuatation.
func (b Vector) Permute(P permutation) {
	if len(b) != len(P) {
		panic("PermuteCols: permutation is of the wrong size\n")
	}
	i := 0
	for i < len(P) {
		if P[i] < 0 {
			i++
			continue
		}

		prev := i
		var next int
		for P[prev] != i {
			next = P[prev]
			b[next], b[prev] = b[prev], b[next]
			P[prev] = -1 - next
			prev = next
		}
		P[prev] = -1 - P[prev]
	}
}

// Scale scales a vector.
func (v Vector) Scale(c float64) {
	for i, _ := range v {
		v[i] *= c
	}
}
