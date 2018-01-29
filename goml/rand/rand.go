// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
// c++ code is written by Prof. Michael Gashler at Univeristy of
// Arkansas, Fayetteville, AR
// http://csce.uark.edu/~mgashler/
package rand

import (
	"math"
)

// This is a 64-bit pseudo-random number generator. This class is superior
// to the standard C rand() method because it is faster, it is consistent
// across platforms, it makes bigger random numbers, and it can draw from
// several useful distributions. The only methods you will use in this class are
// setSeed, next, uniform, and normal. I just left the other methods here
// for completeness.
type Rand struct {
	a, b uint64
}

func NewRand(seed uint64) *Rand {
	var r Rand
	r.SetSeed(seed)
	return &r
}

func (r *Rand) SetSeed(seed uint64) {
	r.b = 0xCA535ACA9535ACB2 + seed
	r.a = 0x6CCF6660A66C35E7 + (seed << 24)
}

// Returns an unsigned pseudo-random 64-bit value
func (r *Rand) next() uint64 {
	r.a = 0x141F2B69*(r.a&0x3ffffffff) + (r.a >> 32)
	r.b = 0xC2785A6B*(r.b&0x3ffffffff) + (r.b >> 32)
	return r.a ^ r.b
}

// Returns a pseudo-random uint from a discrete uniform
// distribution in the range 0 to range-1 (inclusive).
// (This method guarantees the result will be drawn from
// a uniform distribution, whereas doing "next() % range"
// does not guarantee a truly uniform distribution.)
func (r *Rand) Next(domain uint64) uint64 {
	// Use rejection to find a random value in a range that is a
	// multiple of "domain"
	var n uint64 = (0xffffffffffffffff % domain) + 1
	var x uint64

	x = r.next()
	for (x + n) < n {
		x = r.next()
	}

	// Use modulus to return the final value
	return x % domain
}

func (r *Rand) Uniform() float64 {
	return float64(r.next()&0xfffffffffffff) / 4503599627370496.0
}

// Normal returns a random value from a standard normal distribution.
// (To convert it to a random value from an arbitrary normal
// distribution, just multiply the value this returns by the
// deviation (usually lowercase-sigma), then add the mean (usually
// mu).)
func (r *Rand) Normal() float64 {
	var x, y, mag float64
	for mag >= 1.0 || mag == 0 {
		x = r.Uniform()*2 - 1
		y = r.Uniform()*2 - 1
		mag = x*x + y*y
	}
	return y * math.Sqrt(-2.0*math.Log(mag)/mag)
}

// Returns a random value from a categorical distribution
// with the specified vector of category probabilities.
func (r *Rand) Categorical(probabilities []float64) int {
	d := r.Uniform()
	for i := 0; i < len(probabilities); i++ {
		d -= probabilities[i]
		if d < 0 {
			return i
		}
	}
	panic("The probabilities are not normalized!")
}

// Poisson returns a random value from a Poisson distribution
func (r *Rand) Poisson(mu float64) int {
	if mu <= 0 {
		panic("invalid parameter")
	}

	n := 0
	if mu < 30 {
		mu = math.Exp(-mu)
		p := r.Uniform()
		for p >= mu {
			n++
			p *= r.Uniform()
		}
		return n - 1
	} else {
		var u1, u2, x, y float64
		c := 0.767 - 3.36/mu
		b := math.Pi / math.Sqrt(3.0*mu)
		a := b * mu
		if c <= 0 {
			panic("Error generating Poisson deviate")
		}
		k := math.Log(c) - mu - math.Log(b)
		ck1 := 0.0
		var ck2 float64
		for ck1 < 0.5 {
			ck2 = 0.
			for ck2 < 0.5 {
				u1 = r.Uniform()
				x = (a - math.Log(0.1e-18+(1.0-u1)/u1)) / b
				if x > -0.5 {
					ck2 = 1.0
				}
			}
			n = int(x + 0.5)
			u2 = r.Uniform()
			y = 1 + math.Exp(a-b*x)
			ck1 = a - b*x + math.Log(.1e-18+u2/(y*y))
			u2, _ = math.Lgamma(float64(n) + 1.0)
			ck2 = k + float64(n)*math.Log(.1e-18+mu) - u2
			if ck1 <= ck2 {
				ck1 = 1.0
			}
		}
		return n
	}
}
