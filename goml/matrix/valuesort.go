package matrix

type valuesort []struct {
  val float64
  idx int
}

func (v valuesort) Len() int {
  return len(v)
}

func (v valuesort) Less(i, j int) bool {
  return v[i].val > v[j].val
}

func (v valuesort) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}

