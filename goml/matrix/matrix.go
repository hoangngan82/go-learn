// Package matrix implement a mathematical matrix together with some
// metadata. Each element is represented as a float64 (double).
// Matrices are stored in row-major order and all columns have a
// name.
package matrix

import (
  "fmt"
  "bytes"
  "os"
  "strconv"
  "time"
  "bufio"
  "strings"
  "sort"
  "math"
)

const (
  UNKNOWN_VALUE   = -1e308
  TIME_FORMAT     = "2006-01-02 15:04:05"
  EXTRA_NUM_CELL  = 4
  ATTR_NAME       = 1 << 20
  esp             = 1e-15
)

type permutation []int

type matrix []Vector

type Matrix struct {
  matrix
  rows, cols int

  // Metadata
  relation    string

  // attrName[i] stores the attribute name of column[i].
  attrName    []string

  // str_to_enum[i] stores the details of the data in column[i].
  // If column[i] is about a continuous variable or a datetime
  // variable (stored as Unix time) then str_to_enum[i] is an empty
  // map.
  // If column[i] is about a nominal variable then str_to_enum[i] is
  // a map from string value to enumerated value.
  // This is useful when loading data from an arff file.
  str_to_enum [] map[string]int

  // enum_to_str[i] maps ATTR_NAME to data type of column[i].
  // Currently, the code only support REAL, NOMINAL, and DATE data.
  // If column[i] is a nominal variable then enum_to_str[i] maps its
  // enumerated values to its string values.
  enum_to_str [] map[int]string
}

// NewMatrix creates a matrix of size rows*cols and set all element
// to the value val (default = 0).
func NewMatrix (rows, cols int, val ...float64) *Matrix {
  Require (rows > 0 && cols > 0,
    "NewMatrix: cannot generate an empty matrix of size %d-by-%d\n",
    rows, cols)
  var m Matrix
  m.rows = rows
  m.cols = cols

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
  m.matrix = make ([]Vector, rows, rows + EXTRA_NUM_CELL)
  N := len(val)
  if N == 0 {
    for i:= 0; i < rows; i++ {
      m.matrix[i] = make ([]float64, cols, cols + EXTRA_NUM_CELL)
    }
    return &m
  }

  for i:= 0; i < rows; i++ {
    m.matrix[i] = make ([]float64, cols, cols + EXTRA_NUM_CELL)
    for j:= 0; j < cols; j++ {
      m.matrix[i][j] = val[(i*m.cols + j)%N]
    }
  }

  return &m
}

// SetElem sets the element at row i, col j to val.
func (m *Matrix) SetElem(i, j int, val float64) {
  m.matrix[i][j] = val
}

// SetElems sets the elements at rows and cols to val's.
func (m *Matrix) SetElems(rows, cols []int, vals []float64) {
  Require (len(rows) != 0 && len(cols) != 0 && len(vals) != 0,
    "SetElems: empty index: len(rows) = %d, len(cols) = " +
    "%d, len(vals) = %d\n", len(rows), len(cols), len(vals))
  Require (len(rows) == len(cols) && len(cols) == len(vals),
    "SetElems: dimension mismatched: len(rows) = %d, len(cols) = " +
    "%d, len(vals) = %d\n", len(rows), len(cols), len(vals))
  for i, val := range vals {
    m.matrix[rows[i]][cols[i]] = val
  }
}

// GetElem returns the element at row r and column c in the Matrix
func (m *Matrix) GetElem(r, c int) float64 {
  return m.matrix[r][c]
}

// GetCol returns a copy of the col j.
func (m *Matrix) GetCol(j int) Vector {
  Require (j >= 0 && j < m.cols,
    "GetCol: index out of bound: c = %d\n", j)
  c := make ([]float64, m.rows)
  for i:= 0; i < m.cols; i++ {
    c[i] = m.matrix[i][j]
  }
  return c
}

// GetRow wraps a vector around the row i.
func (m *Matrix) GetRow(i int) Vector {
  Require (i >= 0 && i < m.rows,
    "GetCol: index out of bound: r = %d\n", i)
  return m.matrix[i]
}


// String converts a Matrix into a string so that it can be printed
// using fmt.Printf("%v\n", matrix).
func (m *Matrix) String() string {
  var buf bytes.Buffer

  // find the width of a printed row
  numstr := 0
  for j:= 0; j < m.cols; j++ {
    switch m.enum_to_str[j][ATTR_NAME][0] {
    case 'n': // nominal
      numstr += 11
    default:  // real
      numstr += 21
    }
  }
  heline := make([]byte, numstr)
  hdline := make([]byte, numstr)
  for i:= 0; i < len(heline); i++ {
    heline[i] = '='
    hdline[i] = '-'
  }

  buf.Grow((m.rows + 2)*m.cols*21 + m.rows*2 + 2 + 2)
  fmt.Fprintf(&buf, "\n%s\n", heline)
  // print attribute type
  for j:= 0; j < m.cols; j++ {
    switch m.enum_to_str[j][ATTR_NAME][0] {
    case 'n': // nominal
      fmt.Fprintf(&buf, " %10q", "nominal")
    default:  // real
      fmt.Fprintf(&buf, " %20q", m.enum_to_str[j][ATTR_NAME])
    }
  }
  fmt.Fprintf(&buf, "\n")
  // print attribute name
  for j:= 0; j < m.cols; j++ {
    switch m.enum_to_str[j][ATTR_NAME][0] {
    case 'n': // nominal
      if len(m.attrName[j]) > 10 {
        fmt.Fprintf(&buf, " %10s", m.attrName[j][:10])
      } else {
        fmt.Fprintf(&buf, " %10s", m.attrName[j])
      }
    default:  // real
      fmt.Fprintf(&buf, " %20s", m.attrName[j])
    }
  }

  // print data
  fmt.Fprintf(&buf, "\n%s\n", hdline)
  for i:= 0; i < m.rows; i++ {
    for j:= 0; j < m.cols; j++ {
      switch m.enum_to_str[j][ATTR_NAME][0] {
      case 'n': // nominal
        fmt.Fprintf(&buf, " %10d", int(m.matrix[i][j]))
      case 'd': // date
        t := int64(m.matrix[i][j])
        fmt.Fprintf(&buf, " %20s",
          time.Unix(t, 0).Format(TIME_FORMAT))
      default:  // real
        fmt.Fprintf(&buf, " %20.12e", m.matrix[i][j])
      }
    }
    fmt.Fprintf(&buf, "\n")
  }
  fmt.Fprintf(&buf, "%s\n", heline)
  return buf.String()
}

// Rows return the number of rows of a matrix
func (m *Matrix) Rows() int {
  return m.rows
}

// Cols return the number of columns of a matrix
func (m *Matrix) Cols() int {
  return m.cols
}

// Size return the size of a matrix
func (m *Matrix) Size() (r int, c int) {
  return m.rows, m.cols
}

// Transpose returns the transpose of a matrix.
func (m *Matrix) Transpose() *Matrix {
  t := NewMatrix(m.cols, m.rows)

  // actual data
  for i := 0; i < t.rows; i++ {
    for j:= 0; j < t.cols; j++ {
      t.matrix[i][j] = m.matrix[j][i]
    }
  }
  return t
}


// Mul stores the product of the two matracies in the receiver.
func (m *Matrix) Mul(a, b *Matrix, aTranspose, bTranspose bool) {
  cols, rows := a.cols, b.rows
  m.rows, m.cols = a.rows, b.cols
  if aTranspose { cols, m.rows = m.rows, cols }
  if bTranspose { rows, m.cols = m.cols, rows }
  Require (cols == rows, "matrix: Mul: dimension mismatched\n")

  // metadata
  m.relation = "default"
  m.attrName = make ([]string, m.cols, m.cols + EXTRA_NUM_CELL)
  m.str_to_enum = make ([] map[string]int, m.cols, m.cols + EXTRA_NUM_CELL)
  m.enum_to_str = make ([] map[int]string, m.cols, m.cols + EXTRA_NUM_CELL)
  for i:= 0; i < m.cols; i++ {
    m.attrName[i] = fmt.Sprintf("col_%d",i)
    m.enum_to_str[i] = make (map[int]string)
    m.enum_to_str[i][ATTR_NAME] = "real"
  }

  // allocate memory
  m.matrix = make ([]Vector, m.rows, m.rows + EXTRA_NUM_CELL)
  for i := 0; i < m.rows; i++ {
    m.matrix[i] = make ([]float64, m.cols, m.cols + EXTRA_NUM_CELL)
  }

  // compute the product
  if aTranspose {   // a is transposed
    if bTranspose { // a and b are transposed
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.rows; k++ {
            m.matrix[i][j] += a.matrix[k][i]*b.matrix[j][k]
          }
        }
      }
    } else {  // a is tranposed but b is not
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.rows; k++ {
            m.matrix[i][j] += a.matrix[k][i]*b.matrix[k][j]
          }
        }
      }
    }
  } else {  // a is not transposed
    if bTranspose { // a is not transposed but b is
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.cols; k++ {
            m.matrix[i][j] += a.matrix[i][k]*b.matrix[j][k]
          }
        }
      }
    } else { // a and b are not transposed
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.cols; k++ {
            m.matrix[i][j] += a.matrix[i][k]*b.matrix[k][j]
          }
        }
      }
    }
  }
}

func Mul(a, b *Matrix, aTranspose, bTranspose bool) *Matrix {
  var c Matrix
  c.Mul(a, b, aTranspose, bTranspose)
  return &c
}


// AddRows adds more rows to the matrix. All data are wiped out.
func (m *Matrix) AddRows(n int) {
  Require (n > 0, "AddRows: n must be positive")
  oldRow := m.rows
  m.rows += n
  if m.rows <= cap(m.matrix) {
    m.matrix = m.matrix[:m.rows]
  } else {
    temp := make ([]Vector, m.rows, m.rows + EXTRA_NUM_CELL)
    copy (temp, m.matrix)
    m.matrix = temp
  }
  for i:= oldRow; i < m.rows; i++ {
    m.matrix[i] = make([]float64, m.cols, m.cols + EXTRA_NUM_CELL)
  }
}

// AddCols adds more cols to the matrix. All data are wiped out.
func (m *Matrix) AddCols(n int) {
  Require (n > 0, "AddCols: n must be positive")
  oldCol := m.cols
  m.cols += n
  if m.cols <= cap(m.matrix[0]) {
    for i:= 0; i < m.rows; i++ {
      m.matrix[i] = m.matrix[i][:m.cols]
    }
    m.attrName = m.attrName[:m.cols]
    m.enum_to_str = m.enum_to_str[:m.cols]
    m.str_to_enum = m.str_to_enum[:m.cols]
  } else {
    for i:= 0; i < m.rows; i++ {
      temp := make([]float64, m.cols, m.cols + EXTRA_NUM_CELL)
      copy (temp, m.matrix[i])
      m.matrix[i] = temp
    }
  }
  temp := make([]string, m.cols, m.cols + EXTRA_NUM_CELL)
  copy (temp, m.attrName)
  m.attrName = temp
  temps := make([]map[string]int, m.cols, m.cols + EXTRA_NUM_CELL)
  copy (temps, m.str_to_enum)
  m.str_to_enum = temps
  tempe := make([]map[int]string, m.cols, m.cols + EXTRA_NUM_CELL)
  copy (tempe, m.enum_to_str)
  m.enum_to_str = tempe
  for i:= oldCol; i < m.cols; i++ {
    m.attrName[i] = fmt.Sprintf("col_%d",i)
    m.enum_to_str[i] = make(map[int]string)
    m.enum_to_str[i][ATTR_NAME] = "real"
  }
}

// Scale scales all element by the factor.
func (m *Matrix) Scale(c float64) {
  for i:= 0; i < m.rows; i++ {
    for j:= 0; j < m.cols; j++ {
      m.matrix[i][j] *= c
    }
  }
}

// ChangeAttrName changes the names of a list of attributes. If the
// new name is an empty string or is not defined, the old name is
// kept.
func (m *Matrix) ChangeAttrName(newNames []string) {
  for i:= 0; i < m.cols && i < len(newNames); i++ {
    if newNames[i] != "" {
      m.attrName[i] = newNames[i]
    }
  }
}

// GetAttrName returns the name of the specified attribute.
func (m *Matrix) GetAttrName(col int) string {
  return m.attrName[col]
}

// GetAttrType returns the type of the specified attribute.
func (m *Matrix) GetAttrType(col int) string {
  return m.enum_to_str[col][ATTR_NAME]
}

// GetValueName returns the name of the specified value.
func (m *Matrix) GetValueName(col, val int) string {
  if len(m.str_to_enum[col]) == 0 {
    return ""
  }
  return m.enum_to_str[col][val]
}


// SaveARFF save data from a matrix to an ARFF file.
func (m *Matrix) SaveARFF(fileName string) {
  fileio, err := os.Create(fileName)
  Require (err == nil, "SaveARFF: %v\n", err)
  defer fileio.Close()

  file := bufio.NewWriter(fileio)
  file.WriteString("@relation " + m.relation + "\n\n")
  file.Flush()
  for i:= 0; i < m.cols; i++ {
    quote := ""
    for j:= 0; j < len(m.attrName[i]); j++ {
      if m.attrName[i][j] == ' ' {
        quote = "\""
        break
      }
    }
    file.WriteString("@attribute " + quote + m.attrName[i] +
      quote + "\t")
    s := m.enum_to_str[i][ATTR_NAME]
    switch s[0] {
    case 'n': // nominal
      file.WriteString("{" + m.enum_to_str[i][0])
      for j:= 1; j < len(m.str_to_enum[i]); j++ {
        file.WriteString("," + m.enum_to_str[i][j])
      }
      file.WriteString("}\n")
    case 'r': // real
      file.WriteString("real\n")
    default:  // date
      file.WriteString("\"yyyy-MM-dd HH:mm:ss\"\n")
    }
    file.Flush()
  }
  file.WriteString("\n@data\n")
  // write data
  for i:= 0; i < m.rows; i++ {
    for j:= 0; j < m.cols; j++ {
      if j > 0 {
        file.WriteString(",")
      }
      var s string
      switch m.enum_to_str[j][ATTR_NAME][0] {
      case 'n':   // nominal
        s = fmt.Sprintf("%s",m.enum_to_str[j][int(m.matrix[i][j])])
      case 'r':   // real
        s = fmt.Sprintf("%.15e", m.matrix[i][j])
      default:    // date
        t := int64(m.matrix[i][j])
        s = fmt.Sprintf("%s", time.Unix(t, 0).Format(TIME_FORMAT))
      }
      file.WriteString(s)
    }
    file.WriteString("\n")
    file.Flush()
  }
}

// LoadARFF loads data from ARFF file to a matrix
func (m *Matrix) LoadARFF(fileName, timeZone string) {
  fileio, err := os.Open(fileName)
  Require (err == nil, "LoadARFF: %v\n", err)
  defer fileio.Close()

  // metadata
  m.attrName = make ([]string, 0, EXTRA_NUM_CELL)
  m.str_to_enum = make ([] map[string]int, 0, EXTRA_NUM_CELL)
  m.enum_to_str = make ([] map[int]string, 0, EXTRA_NUM_CELL)

  attrFile := bufio.NewReader(fileio)
  var line string

  // read attributes' names and data types.
  lineNum := 0  // current line
  m.cols = 0
  for err == nil {
    line, err = attrFile.ReadString('\n')
    if err != nil {
      break
    }
    lineNum++
    line = line[:len(line)-1]
    s := Split(line, "\t ", 0, 3)
    if len(s) == 0 { continue }
    switch strings.ToLower(s[0]) {
    case "@relation":
      m.relation = s[1]
    case "@attribute":
      m.attrName = append(m.attrName, s[1])
      m.enum_to_str = append(m.enum_to_str, make(map[int]string))
      m.str_to_enum = append(m.str_to_enum, make(map[string]int))
      switch strings.ToLower(string(s[2][0])) {
      case "{": // nominal
        m.enum_to_str[m.cols][ATTR_NAME] = "nominal"
        sn := s[2][1:len(s[2]) - 1]
        n  := Split(sn, ",", 0)
        for i := 0; i < len(n); i++ {
          m.str_to_enum[m.cols][n[i]] = i
          m.enum_to_str[m.cols][i] = n[i]
        }
      case "i": // integer
        fallthrough
      case "r": // real
        fallthrough
      case "n": // numeric
        m.enum_to_str[m.cols][ATTR_NAME] = "real"
      case "s": // string
        panic("LoadARFF: string data is not supported")
      case "d": // date
        s := Split(line, "\t ", 0, 4)
        Require (s[3] == "yyyy-MM-dd HH:mm:ss" ||
          s[3] == "yyyy-MM-ddTHH:mm:ss",
          "LoadARFF: only date formats '%s' and '%s' are supported\n",
          "yyyy-MM-dd HH:mm:ss", "yyyy-MM-ddTHH:mm:ss" )
        m.enum_to_str[m.cols][ATTR_NAME] = "date"
      default:
        panic(fmt.Sprintf(
          "Attribute %s: data type %q is not supported\n",
          s[1], s[2]))
      }
      m.cols++
    case "@data":
      err = fmt.Errorf("done with reading attribute")
    default:
    }
  }

  // read data
  m.matrix = make ([]Vector, 0, EXTRA_NUM_CELL)
  err = nil
  row := 0
  for err == nil {
    line, err = attrFile.ReadString('\n')
    if err != nil {
      break
    }
    lineNum++
    line = line[:len(line)-1]
    if len(line) == 0 || line[0] == '%' { continue }

    s := Split(line, "\t ,", 0)
    Require (len(s) == m.cols,
      "LoadARFF: %s: wrong number of attributes on line %d\n",
      fileName, lineNum )
    m.matrix = append (m.matrix, make ([]float64, m.cols, m.cols +
      EXTRA_NUM_CELL))
    for j := 0; j < len(s); j++ {
      if s[j] == "?" {
        m.matrix[row][j] = UNKNOWN_VALUE
      } else {
        switch m.enum_to_str[j][ATTR_NAME][0] {
        case 'n': // nominal
          m.matrix[row][j] =
            float64(m.str_to_enum[j][s[j]])
        case 'r': // real
          number, parseErr := strconv.ParseFloat(s[j], 64)
          Require (parseErr == nil || parseErr != strconv.ErrSyntax,
          "LoadARFF: %s: data at column %d on line %d is not a valid real number\n",
            fileName, j, lineNum)
          m.matrix[row][j] = number
        case 'd': // date
          t, parseErr := time.Parse (time.RFC3339,
            strings.Replace(s[j], " ", "T", 1) + timeZone )
          Require (parseErr == nil,
          "LoadARFF: %s: data at column %d on line %d is not a valid date\n",
            fileName, j, lineNum )
          m.matrix[row][j] = float64 (t.Unix())
        default:  // string is not implemented
        }
      }
    }
    row++
  }
  m.rows = row
}

// SubMatrix copy the content of a submatrix of a matrix determined
// by the list of column indices and row indices. This operation
// allows duplicating rows and columns.
func (m *Matrix) SubMatrix(s *Matrix, rows, cols []int) {
  if len(rows) != s.rows || len(cols) != s.cols {
    s = NewMatrix(len(rows), len(cols))
  }
  for i:= 0; i < len(rows); i++ {
    for j:= 0; j < len(cols); j++ {
      s.matrix[i][j] = m.matrix[rows[i]][cols[j]]
    }
  }
}

// SwapRows swaps two rows in the matrix.
func (m *Matrix) SwapRows(r1, r2 int) {
  m.matrix[r1], m.matrix[r2] = m.matrix[r2], m.matrix[r1]
}

// SwapCols swaps two columns in the matrix (including metadata).
func (m *Matrix) SwapCols(c1, c2 int) {
  m.attrName[c1], m.attrName[c2] = m.attrName[c2], m.attrName[c1]
  m.enum_to_str[c1], m.enum_to_str[c2] = m.enum_to_str[c2], m.enum_to_str[c1]
  m.str_to_enum[c1], m.str_to_enum[c2] = m.str_to_enum[c2], m.str_to_enum[c1]
  for i:= 0; i < m.rows; i++ {
    m.matrix[i][c1], m.matrix[i][c2] = m.matrix[i][c2], m.matrix[i][c1]
  }
}

// axb returns the vector A*x + sign*b where sign is in {-1, 1}.
func (a *Matrix) axb(x, b Vector, sign int) Vector {
  Require(a.cols == len(x) && a.rows == len(b),
  "Axb: dimension mismatched: a.cols == len(x) && a.rows == len(b)\n")
  v := NewVector(a.rows)
  if sign < 0 {
    for i:= 0; i < len(v); i++ {
      for j:= 0; j < a.cols; j++ {
        v[i] += a.matrix[i][j]*x[j]
      }
      v[i] -= b[i]
    }
  } else {
    for i:= 0; i < len(v); i++ {
      for j:= 0; j < a.cols; j++ {
        v[i] += a.matrix[i][j]*x[j]
      }
      v[i] += b[i]
    }
  }
  return v
}

// Axpb returns the vector A*x + b.
func Axpb(A *Matrix, x, b Vector) Vector {
  return A.axb(x, b, 1)
}

// Axmb returns the vector A*x - b.
func Axmb(A *Matrix, x, b Vector) Vector {
  return A.axb(x, b, -1)
}

// PermuteCols change the columns of the matrix due to a permutation.
func (m *Matrix) PermuteCols(P permutation) {
  if (m.cols != len(P)) {
    panic("PermuteCols: permutation is of the wrong size\n")
  }
  i := 0
  for i < len(P) {
    for P[i] != i {
      j := P[i]
      k := P[j]
      m.SwapCols(j, k)
      P[i], P[j] = P[j], P[i]
    }
    i++
  }
}

// PermuteRows change the rows of the matrix due to a permutation.
func (m *Matrix) PermuteRows(P permutation) {
  if (m.rows != len(P)) {
    panic("PermuteCols: permutation is of the wrong size\n")
  }
  i := 0
  for i < len(P) {
    for P[i] != i {
      j := P[i]
      k := P[j]
      m.SwapRows(j, k)
      P[i], P[j] = P[j], P[i]
    }
    i++
  }
}


// LeastSquare solves the solution to the least square problem with
// assumption that the norm of the solution is minimum.
// We use the pivoting Householder QR method.
func (m *Matrix) LeastSquare(b Vector) Vector {
  Require(m.rows >= m.cols,
    "LeastSquare: expected at least as many rows as columns\n")
  Require(m.rows == len(b), "LeastSquare: dimension mismatched\n")
  var x Vector = NewVector(m.cols)

  var P permutation
  rank := m.QR(b, &P)

  // backward substitution
  i := rank - 1
  k := P[i]
  x[k] = b[i]/m.matrix[i][k]
  var l int
  for i:= rank-2; i >= 0; i-- {
    k = P[i]
    x[k] = b[i]
    for j:= i+1; j < rank; j++ {
      l = P[j]
      x[k] -= x[l]*m.matrix[i][l]
    }
    x[k] /= m.matrix[i][k]
  }

  return x
}


// QR perform QR factorization on the matrix m and return the rank of
// m. After QR is done, m will be R, b will be Q*b, and *p maps the
// new columns' positions to their original ones. For example,
// (*p)[0] is the original column of column 0, i.e, column (*p)[0] of
// the matrix R returned by this QR code is the column 0 of the
// matrix R after doing QR factoriation with column pivoting.
func (m *Matrix) QR(b Vector, p *permutation) int {
  var P permutation = make([]int, len(b))

  // pre-sort rows by ascending order
  rowNorms := make(valuesort, len(b))
  for i:= 0; i < len(b); i++ {
    rowNorms[i].val = m.GetRow(i).Norm(0)
    rowNorms[i].idx = i
  }
  sort.Sort(rowNorms)
  for i:= 0; i < len(b); i++ {
    P[i] = rowNorms[i].idx
  }

  // sort the rows based on sorted norms, after this P will become
  // the identity permuation.
  m.PermuteRows(P)

  for i:= 0; i < len(b); i++ {
    P[i] = rowNorms[i].idx
  }
  b.Permute(P)

  // now perform Householder with column pivoting
  // We will perform column pivoting implicitly using a permutation.

  // Identity permutation on columns.
  *p = make([]int, m.cols)
  P = *p
  for i:= 0; i < len(P); i++ {
    P[i] = i
  }

  // Householder step on both A and b.
  maxColNorm := float64(0)
  maxColIndex := 0
  colNorm := maxColNorm
  vk := NewVector(m.rows)
  x  := NewVector(m.rows)
  var l int
  for k:= 0; k < m.cols; k++ {
    // determine column with largest 2-norm
    maxColNorm = 0
    for j:= k; j < m.cols; j++ {
      colNorm = 0
      l = P[j]
      for i:= k; i < m.rows; i++ {
        colNorm += m.matrix[i][l]*m.matrix[i][l]
      }
      Require(!math.IsNaN(colNorm) && !math.IsInf(colNorm, 0),
        "LeastSquare: the norm of column %d is not a number.\n", P[j])
      if colNorm > maxColNorm {
        maxColNorm = colNorm
        maxColIndex = j
      }
    }

    a00 := m.matrix[0][P[0]]
    if maxColNorm < a00*esp*esp*a00 {
      return k
    }
    // swap columns i and maxColIndex
    P[k], P[maxColIndex] = P[maxColIndex], P[k]

    // Get the k-th column vector
    l = P[k]
    for j:= k; j < m.rows; j++ {
      vk[j] = m.matrix[j][l]
    }

    colNorm = vk[k:].Norm(2)
    if vk[k] < 0 {
      vk[k] -= colNorm
    } else {
      vk[k] += colNorm
    }

    colNorm = vk[k:].Norm(2)
    colNorm = float64(1)/colNorm
    vk[k:].Scale(colNorm)

    // A[k:m, k:n] -= 2vk(vk*A[k:m, k:n])
    for j:= k; j < m.cols; j++ {
      l = P[j]
      for i:= k; i < m.rows; i++ {
        x[i] = m.matrix[i][l]
      }
      colNorm = 2*x[k:].Dot(vk[k:])
      for i:= k; i < m.rows; i++ {
        m.matrix[i][l] -= colNorm*vk[i]
      }
    }

    // compute Q*b
    colNorm = 2*b[k:].Dot(vk[k:])
    for i:= k; i < m.rows; i++ {
      b[i] -= colNorm*vk[i]
    }
  }
  return m.cols
}
