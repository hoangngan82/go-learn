// Package vector implement a mathematical vector together with some
// metadata. Each element is represented as a float64 (double).
// Matrices are stored in column-major order and all columns have a
// name.
package matrix

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
)

type Vector []float64

// ToMatrix wraps a 1-by-n matrix around the underlying data of a
// vector.
func (v Vector) ToMatrix(dim ...int) *Matrix {
	if len(dim) == 0 {
		return NewMatrix(1, len(v), v)
	} else {
		return NewMatrix(dim[0], len(v)/dim[0], v)
	}
}

// NewVector wraps a vector around vals.
func NewVector(size int, vals []float64) Vector {
	Require(len(vals) == 0 || len(vals) == size,
		"NewVector: require len(vals) == 0 || len(vals) == size\n")
	if len(vals) > 0 {
		return vals
	}

	return make(Vector, size)
}

// String converts a Vector into a string so that it can be printed
// using fmt.Printf("%v\n", vector).
func (v Vector) String() string {
	var buf bytes.Buffer
	buf.Grow(len(v)*22 + 3 + 2)
	fmt.Fprintf(&buf, "\n[\n")
	for i := 0; i < len(v); i++ {
		fmt.Fprintf(&buf, " %20.12e\n", v[i])
	}
	fmt.Fprintf(&buf, "]\n")
	return buf.String()
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
	s := make(Vector, len(a))
	copy(s, a)
	s.add(b, -1)
	return s
}

// Add returns the sum a + b.
func (a Vector) Add(b Vector) Vector {
	s := make(Vector, len(a))
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

	for i := 0; i < len(P); i++ {
		P[i] = -1 - P[i]
	}
}

// Fill fills a vector with value.
func (v Vector) Fill(val float64) Vector {
	for i, _ := range v {
		v[i] = val
	}
	return v
}

// Copy just a wrapper of the built-in function copy.
func (v Vector) Copy(m Vector) {
	copy(v, m)
}

// Scale scales a vector.
func (v Vector) Scale(c float64) Vector {
	for i, _ := range v {
		v[i] *= c
	}
	return v
}

// Random set a random normal value (0, 1) for each element of the
// vector v.
func (v Vector) Random(seed ...int64) {
	var s int64 = 1982
	if len(seed) > 0 {
		s = seed[0]
	}
	r := rand.New(rand.NewSource(s))
	for i := 0; i < len(v); i++ {
		v[i] = r.NormFloat64()
	}
}
