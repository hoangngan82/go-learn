package matrix

// Split splits the s string into tokens separated by any of the characters
// in the sep string and return the number of tokens. The location of
// each token in the string s is stored in the slice m. The optional
// parameter n determine the index where the search starts, the
// default value is 0. Split ignores all sep inside quoted string.
func Split(s, sep string, idx0 int, n ...int) []string {
  // quote marks the apperance of " (quote[0]) and '
  quote := [2]bool{false, false}
  var total int = 0
  var i int = idx0
  var N int = int(1 << 30)
  if len(n) > 0 {
    N = n[0]
  }
  m := make([]string, 0, 10)
  for i < len(s) && total < N {
    foundSep := false
    start := i
    end   := i
    for k := i; k < len(s); k++ {
      // doubly-quoted string
      if !quote[1] && s[k] == '"' {
        quote[0] = !quote[0]
        if quote[0] {
          start++
        } else {
          foundSep = true
          if end > start {
            m = append(m, s[start:end])
            total++
          }
          i = k + 1
        }
      }

      // singly-quoted string
      if !quote[0] && s[k] == '\x27' {
        quote[1] = !quote[1]
        if quote[1] {
          start++
        } else {
          foundSep = true
          if end > start {
            m = append(m, s[start:end])
            total++
          }
          i = k + 1
        }
      }

      // if not start with quote, look for sep
      if !quote[0] && !quote[1] {
        for j := 0; j < len(sep); j++ {
          if s[k] == sep[j] {
            foundSep = true
            if end > start {
              m = append(m, s[start:end])
              total++
            }
            i = k + 1
            break
          }
        }
      }
      if foundSep { break }
      end++
      if end == len(s) {
        i = end
        if end > start {
          m = append(m, s[start:end])
          total++
        }
      }
    }
  }
  return m
}
