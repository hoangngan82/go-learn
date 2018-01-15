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
)

const (
  UNKNOWN_VALUE   = -1e308
  TIME_FORMAT     = "2006-01-02 15:04:05"
  EXTRA_NUM_CELL  = 4
  ATTR_NAME       = 1 << 20
)

type Matrix struct {
  data        [][]float64
  rows, cols  int

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
  m.data = make ([][]float64, rows, rows + EXTRA_NUM_CELL)
  N := len(val)
  if N == 0 {
    for i:= 0; i < rows; i++ {
      m.data[i] = make ([]float64, cols, cols + EXTRA_NUM_CELL)
    }
    return &m
  }

  for i:= 0; i < rows; i++ {
    m.data[i] = make ([]float64, cols, cols + EXTRA_NUM_CELL)
    for j:= 0; j < cols; j++ {
      m.data[i][j] = val[(i*m.cols + j)%N]
    }
  }

  return &m
}

// SetElem sets the element at row i, col j to val.
func (m *Matrix) SetElem(i, j int, val float64) {
  m.data[i][j] = val
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
    m.data[rows[i]][cols[i]] = val
  }
}

// GetElem returns the element at row r and column c in the Matrix
func (m *Matrix) GetElem(r, c int) float64 {
  return m.data[r][c]
}

// GetCol returns a copy of the col j.
func (m *Matrix) GetCol(j int) *Vector {
  Require (j >= 0 && j < m.cols,
    "GetCol: index out of bound: c = %d\n", j)
  var c Vector
  c.size = m.rows
  c.data = make ([]float64, c.size)
  for i:= 0; i < m.cols; i++ {
    c.data[i] = m.data[i][j]
  }
  return &c
}

// GetRow wraps a vector around the row i.
func (m *Matrix) GetRow(i int) *Vector {
  Require (i >= 0 && i < m.rows,
    "GetCol: index out of bound: r = %d\n", i)
  var v Vector
  v.size = m.cols
  v.data = m.data[i]
  //v.data = make ([]float64, m.cols)
  //copy (v.data, m.data[i])
  return &v
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
        fmt.Fprintf(&buf, " %10d", int(m.data[i][j]))
      case 'd': // date
        t := int64(m.data[i][j])
        fmt.Fprintf(&buf, " %20s",
          time.Unix(t, 0).Format(TIME_FORMAT))
      default:  // real
        fmt.Fprintf(&buf, " %20.12e", m.data[i][j])
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
  var t Matrix
  t.rows, t.cols = m.cols, m.rows

  // metadata
  t.relation = "default"
  t.attrName = make ([]string, t.cols, t.cols + EXTRA_NUM_CELL)
  t.str_to_enum = make ([] map[string]int, t.cols, t.cols + EXTRA_NUM_CELL)
  t.enum_to_str = make ([] map[int]string, t.cols, t.cols + EXTRA_NUM_CELL)
  for i:= 0; i < t.cols; i++ {
    t.attrName[i] = fmt.Sprintf("col_%d",i)
    t.enum_to_str[i] = make (map[int]string)
    t.enum_to_str[i][ATTR_NAME] = "real"
  }

  // actual data
  t.data = make ([][]float64, t.rows, t.rows + EXTRA_NUM_CELL)
  for i := 0; i < t.rows; i++ {
    t.data[i] = make ([]float64, t.cols, t.cols + EXTRA_NUM_CELL)
    for j:= 0; j < t.cols; j++ {
      t.data[i][j] = m.data[j][i]
    }
  }
  return &t
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
  m.data = make ([][]float64, m.rows, m.rows + EXTRA_NUM_CELL)
  for i := 0; i < m.rows; i++ {
    m.data[i] = make ([]float64, m.cols, m.cols + EXTRA_NUM_CELL)
  }

  // compute the product
  if aTranspose {   // a is transposed
    if bTranspose { // a and b are transposed
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.rows; k++ {
            m.data[i][j] += a.data[k][i]*b.data[j][k]
          }
        }
      }
    } else {  // a is tranposed but b is not
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.rows; k++ {
            m.data[i][j] += a.data[k][i]*b.data[k][j]
          }
        }
      }
    }
  } else {  // a is not transposed
    if bTranspose { // a is not transposed but b is
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.cols; k++ {
            m.data[i][j] += a.data[i][k]*b.data[j][k]
          }
        }
      }
    } else { // a and b are not transposed
      for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
          for k := 0; k < a.cols; k++ {
            m.data[i][j] += a.data[i][k]*b.data[k][j]
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
  if m.rows <= cap(m.data) {
    m.data = m.data[:m.rows]
  } else {
    temp := make ([][]float64, m.rows, m.rows + EXTRA_NUM_CELL)
    copy (temp, m.data)
    m.data = temp
  }
  for i:= oldRow; i < m.rows; i++ {
    m.data[i] = make([]float64, m.cols, m.cols + EXTRA_NUM_CELL)
  }
}

// AddCols adds more cols to the matrix. All data are wiped out.
func (m *Matrix) AddCols(n int) {
  Require (n > 0, "AddCols: n must be positive")
  oldCol := m.cols
  m.cols += n
  if m.cols <= cap(m.data[0]) {
    for i:= 0; i < m.rows; i++ {
      m.data[i] = m.data[i][:m.cols]
    }
    m.attrName = m.attrName[:m.cols]
    m.enum_to_str = m.enum_to_str[:m.cols]
    m.str_to_enum = m.str_to_enum[:m.cols]
  } else {
    for i:= 0; i < m.rows; i++ {
      temp := make([]float64, m.cols, m.cols + EXTRA_NUM_CELL)
      copy (temp, m.data[i])
      m.data[i] = temp
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
      m.data[i][j] *= c
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
        s = fmt.Sprintf("%s",m.enum_to_str[j][int(m.data[i][j])])
      case 'r':   // real
        s = fmt.Sprintf("%.15e", m.data[i][j])
      default:    // date
        t := int64(m.data[i][j])
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
  m.data = make ([][]float64, 0, EXTRA_NUM_CELL)
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
    m.data = append (m.data, make ([]float64, m.cols, m.cols +
      EXTRA_NUM_CELL))
    for j := 0; j < len(s); j++ {
      if s[j] == "?" {
        m.data[row][j] = UNKNOWN_VALUE
      } else {
        switch m.enum_to_str[j][ATTR_NAME][0] {
        case 'n': // nominal
          m.data[row][j] =
            float64(m.str_to_enum[j][s[j]])
        case 'r': // real
          number, parseErr := strconv.ParseFloat(s[j], 64)
          Require (parseErr == nil || parseErr != strconv.ErrSyntax,
          "LoadARFF: %s: data at column %d on line %d is not a valid real number\n",
            fileName, j, lineNum)
          m.data[row][j] = number
        case 'd': // date
          t, parseErr := time.Parse (time.RFC3339,
            strings.Replace(s[j], " ", "T", 1) + timeZone )
          Require (parseErr == nil,
          "LoadARFF: %s: data at column %d on line %d is not a valid date\n",
            fileName, j, lineNum )
          m.data[row][j] = float64 (t.Unix())
        default:  // string is not implemented
        }
      }
    }
    row++
  }
  m.rows = row
}

// SubMatrix returns a reference to a rectangular subarea.
func (m *Matrix) SubMatrix(topLeft, bottomRight [2]int) *Matrix {
  // slice everything
  var t Matrix
  t.cols = bottomRight[1] - topLeft[1]
  t.rows = bottomRight[0] - topLeft[0]
  t.relation = m.relation
  t.attrName = m.attrName[topLeft[1]:bottomRight[1]]
  t.enum_to_str = m.enum_to_str[topLeft[1]:bottomRight[1]]
  t.str_to_enum = m.str_to_enum[topLeft[1]:bottomRight[1]]
  t.data = m.data[topLeft[0]:bottomRight[0]]
  for i:= 0; i < t.rows; i++ {
    t.data[i] = m.data[i][topLeft[1]:bottomRight[1]]
  }
  return &t
}

// SwapRows swaps two rows in the matrix.
func (m *Matrix) SwapRows(r1, r2 int) {
  m.data[r1], m.data[r2] = m.data[r2], m.data[r1]
}

// SwapCols swaps two columns in the matrix (including metadata).
func (m *Matrix) SwapCols(c1, c2 int) {
  m.attrName[c1], m.attrName[c2] = m.attrName[c2], m.attrName[c1]
  m.enum_to_str[c1], m.enum_to_str[c2] = m.enum_to_str[c2], m.enum_to_str[c1]
  m.str_to_enum[c1], m.str_to_enum[c2] = m.str_to_enum[c2], m.str_to_enum[c1]
  for i:= 0; i < m.rows; i++ {
    m.data[i][c1], m.data[i][c2] = m.data[i][c2], m.data[i][c1]
  }
}

// axb returns the vector A*x + sign*b.
func (a *Matrix) axb(x, b *Vector, sign int) *Vector {
  Require(a.cols == x.size && a.rows == b.size,
  "Axb: dimension mismatched: a.cols == x.size && a.rows == b.size\n")
  axb := NewVector(a.rows)
  if sign < 0 {
    for i:= 0; i < axb.size; i++ {
      for j:= 0; j < a.cols; j++ {
        axb.data[i] += a.data[i][j]*x.data[j] - b.data[j]
      }
    }
  } else {
    for i:= 0; i < axb.size; i++ {
      for j:= 0; j < a.cols; j++ {
        axb.data[i] += a.data[i][j]*x.data[j] + b.data[j]
      }
    }
  }
  return axb
}

// Axpb returns the vector A*x + b.
func Axpb(A *Matrix, x, b *Vector) *Vector {
  return A.axb(x, b, 1)
}

// Axmb returns the vector A*x + b.
func Axmb(A *Matrix, x, b *Vector) *Vector {
  return A.axb(x, b, -1)
}
