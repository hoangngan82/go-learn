// Package matrix implement a mathematical matrix together with some
// metadata. Each element is represented as a float64 (double).
// Matrices are stored in row-major order and all columns have a
// name.
package matrix

import (
	"../rand"
	"bufio"
	"bytes"
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	UNKNOWN_VALUE  = -1e308
	TIME_FORMAT    = "2006-01-02 15:04:05"
	ATTR_NAME      = 1 << 20
	EXTRA_NUM_CELL = 10
	ESP            = 1e-15
)

type permutation []int

type matrix []Vector

type Matrix struct {
	data Vector
	matrix
	rows, cols int

	// Metadata
	relation string

	// attrName[i] stores the attribute name of column[i].
	attrName []string

	// str_to_enum[i] stores the details of the data in column[i].
	// If column[i] is about a continuous variable or a datetime
	// variable (stored as Unix time) then str_to_enum[i] is an empty
	// map.
	// If column[i] is about a nominal variable then str_to_enum[i] is
	// a map from string value to enumerated value.
	// This is useful when loading data from an arff file.
	str_to_enum []map[string]int

	// enum_to_str[i] maps ATTR_NAME to data type of column[i].
	// Currently, the code only support REAL, NOMINAL, and DATE data.
	// If column[i] is a nominal variable then enum_to_str[i] maps its
	// enumerated values to its string values.
	enum_to_str []map[int]string
}

// NewMatrix creates a matrix of size rows*cols and set all element
// to the value val (default = 0).
func NewMatrix(rows, cols int, val []float64) *Matrix {
	Require(rows > 0 && cols > 0,
		"NewMatrix: cannot generate an empty matrix of size %d-by-%d\n",
		rows, cols)
	Require(len(val) == 0 || len(val) == rows*cols,
		"NewMatrix: require len(val) = 0 || len(val) = rows*cols\n")
	var m Matrix
	m.rows = rows
	m.cols = cols

	// metadata
	m.relation = "default"
	m.attrName = make([]string, m.cols)
	m.str_to_enum = make([]map[string]int, m.cols)
	m.enum_to_str = make([]map[int]string, m.cols)
	for i := 0; i < m.cols; i++ {
		m.attrName[i] = fmt.Sprintf("col_%d", i)
		m.enum_to_str[i] = make(map[int]string)
		m.enum_to_str[i][ATTR_NAME] = "real"
	}

	// actual data
	m.matrix = make([]Vector, m.rows)
	if len(val) > 0 {
		m.data = val
	} else {
		m.data = make(Vector, m.rows*m.cols)
	}

	for i := 0; i < m.rows; i++ {
		m.matrix[i] = m.data[i*m.cols : (i+1)*m.cols]
	}

	return &m
}

// Equal test if two matrices are equal.
func (m *Matrix) Equal(b *Matrix, tol ...float64) bool {
	if m.cols != b.cols || m.rows != b.rows {
		return false
	}
	esp := ESP
	if len(tol) > 0 {
		esp = tol[0]
	}
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if math.Abs(m.matrix[i][j]-b.matrix[i][j]) > esp {
				return false
			}
		}
	}
	return true
}

// SetElem sets the element at row i, col j to val.
func (m *Matrix) SetElem(i, j int, val float64) {
	m.matrix[i][j] = val
}

// FillRow fills a row with the value val.
func (m *Matrix) FillRow(r int, val float64) *Matrix {
	Require(-m.rows <= r && r < m.rows,
		"Fillrol: index out of bound: require %d <= r = %d < %d\n",
		-m.rows, r, m.rows)
	if r < 0 {
		r += m.rows
	}
	for i := 0; i < m.cols; i++ {
		m.matrix[r][i] = val
	}
	return m
}

// FillCol fills a column with the value val.
func (m *Matrix) FillCol(c int, val float64) *Matrix {
	Require(-m.cols <= c && c < m.cols,
		"FillCol: index out of bound: require %d <= c = %d < %d\n",
		-m.cols, c, m.cols)
	if c < 0 {
		c += m.cols
	}
	for i := 0; i < m.rows; i++ {
		m.matrix[i][c] = val
	}
	return m
}

// Fill tiles the vector m.data by the vector vals.
func (m *Matrix) Fill(vals []float64) *Matrix {
	N := len(m.data) / len(vals)
	if N*len(vals) < len(m.data) {
		N++
	}
	for i := 0; i < N; i++ {
		copy(m.data[i*len(vals):], vals)
	}
	return m
}

// GetElem returns the element at row r and column c in the Matrix
func (m *Matrix) GetElem(r, c int) float64 {
	return m.matrix[r][c]
}

// Col returns a copy of the col j.
func (m *Matrix) Col(j int) Vector {
	Require(j >= 0 && j < m.cols,
		"Col: index out of bound: c = %d\n", j)
	c := make([]float64, m.rows)
	for i := 0; i < m.rows; i++ {
		c[i] = m.matrix[i][j]
	}
	return c
}

// Row wraps a vector around the row i.
func (m *Matrix) Row(i int) Vector {
	Require(i >= 0 && i < m.rows,
		"Col: index out of bound: r = %d\n", i)
	return m.matrix[i]
}

// String converts a Matrix into a string so that it can be printed
// using fmt.Printf("%v\n", matrix).
func (m *Matrix) String() string {
	var buf bytes.Buffer

	// find the width of a printed row
	numstr := 0
	for j := 0; j < m.cols; j++ {
		switch m.enum_to_str[j][ATTR_NAME][0] {
		case 'n': // nominal
			numstr += 11
		default: // real
			numstr += 21
		}
	}
	heline := make([]byte, numstr)
	hdline := make([]byte, numstr)
	for i := 0; i < len(heline); i++ {
		heline[i] = '='
		hdline[i] = '-'
	}

	buf.Grow((m.rows+2)*m.cols*21 + m.rows*2 + 2 + 2)
	fmt.Fprintf(&buf, "\n%s\n", heline)
	// print attribute type
	for j := 0; j < m.cols; j++ {
		switch m.enum_to_str[j][ATTR_NAME][0] {
		case 'n': // nominal
			fmt.Fprintf(&buf, " %10q", "nominal")
		default: // real
			fmt.Fprintf(&buf, " %20q", m.enum_to_str[j][ATTR_NAME])
		}
	}
	fmt.Fprintf(&buf, "\n")
	// print attribute name
	for j := 0; j < m.cols; j++ {
		switch m.enum_to_str[j][ATTR_NAME][0] {
		case 'n': // nominal
			if len(m.attrName[j]) > 10 {
				fmt.Fprintf(&buf, " %10s", m.attrName[j][:10])
			} else {
				fmt.Fprintf(&buf, " %10s", m.attrName[j])
			}
		default: // real
			fmt.Fprintf(&buf, " %20s", m.attrName[j])
		}
	}

	// print data
	fmt.Fprintf(&buf, "\n%s\n", hdline)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			switch m.enum_to_str[j][ATTR_NAME][0] {
			case 'n': // nominal
				fmt.Fprintf(&buf, " %10d", int(m.matrix[i][j]))
			case 'd': // date
				t := int64(m.matrix[i][j])
				fmt.Fprintf(&buf, " %20s",
					time.Unix(t, 0).Format(TIME_FORMAT))
			default: // real
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
	t := NewMatrix(m.cols, m.rows, nil)

	// actual data
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			t.matrix[i][j] = m.matrix[j][i]
		}
	}
	return t
}

// Mul stores the product of the two matracies in the receiver.
func (m *Matrix) Mul(a, b *Matrix, aTranspose, bTranspose bool) *Matrix {
	cols, rows := a.cols, b.rows
	m.rows, m.cols = a.rows, b.cols
	if aTranspose {
		cols, m.rows = m.rows, cols
	}
	if bTranspose {
		rows, m.cols = m.cols, rows
	}
	Require(cols == rows, "matrix: Mul: dimension mismatched\n")

	// If the underlying data is of correct size, do not allocate new
	// memory.
	if len(m.data) == m.cols*m.rows {
		// metadata
		m.relation = "default"
		m.attrName = make([]string, m.cols)
		m.str_to_enum = make([]map[string]int, m.cols)
		m.enum_to_str = make([]map[int]string, m.cols)
		for i := 0; i < m.cols; i++ {
			m.attrName[i] = fmt.Sprintf("col_%d", i)
			m.enum_to_str[i] = make(map[int]string)
			m.enum_to_str[i][ATTR_NAME] = "real"
		}

		// reassign row index
		m.matrix = make([]Vector, m.rows)
		for i := 0; i < m.rows; i++ {
			m.matrix[i] = m.data[i*m.cols : (i+1)*m.cols]
		}
	} else {
		*m = *NewMatrix(m.rows, m.cols, nil)
	}

	// compute the product
	if aTranspose { // a is transposed
		if bTranspose { // a and b are transposed
			for i := 0; i < m.rows; i++ {
				for j := 0; j < m.cols; j++ {
					m.matrix[i][j] = 0.0
					for k := 0; k < a.rows; k++ {
						m.matrix[i][j] += a.matrix[k][i] * b.matrix[j][k]
					}
				}
			}
		} else { // a is tranposed but b is not
			for i := 0; i < m.rows; i++ {
				for j := 0; j < m.cols; j++ {
					m.matrix[i][j] = 0.0
					for k := 0; k < a.rows; k++ {
						m.matrix[i][j] += a.matrix[k][i] * b.matrix[k][j]
					}
				}
			}
		}
	} else { // a is not transposed
		if bTranspose { // a is not transposed but b is
			for i := 0; i < m.rows; i++ {
				for j := 0; j < m.cols; j++ {
					m.matrix[i][j] = 0.0
					for k := 0; k < a.cols; k++ {
						m.matrix[i][j] += a.matrix[i][k] * b.matrix[j][k]
					}
				}
			}
		} else { // a and b are not transposed
			for i := 0; i < m.rows; i++ {
				for j := 0; j < m.cols; j++ {
					m.matrix[i][j] = 0.0
					for k := 0; k < a.cols; k++ {
						m.matrix[i][j] += a.matrix[i][k] * b.matrix[k][j]
					}
				}
			}
		}
	}
	return m
}

func Mul(a, b *Matrix, aTranspose, bTranspose bool) *Matrix {
	var c Matrix
	c.Mul(a, b, aTranspose, bTranspose)
	return &c
}

// AddRows adds more rows to the matrix. Old data are kept intact.
func (m *Matrix) AddRows(n int) *Matrix {
	Require(n > 0, "AddRows: n must be positive")
	m.rows += n
	m.matrix = make([]Vector, m.rows)
	temp := make(Vector, m.rows*m.cols)
	copy(temp, m.data)
	m.data = temp
	for i := 0; i < m.rows; i++ {
		m.matrix[i] = m.data[i*m.cols : (i+1)*m.cols]
	}
	return m
}

// AddCols adds more cols to the matrix. All data are kept intact.
// If you don't need to keep old data, use NewMatrix instead
func (m *Matrix) AddCols(n int) *Matrix {
	Require(n > 0, "AddCols: n must be positive")
	oldCol := m.cols
	m.cols += n
	temp := make(Vector, m.rows*m.cols)
	for i := 0; i < m.rows; i++ {
		copy(temp[i*m.cols:], m.data[i*oldCol:(i+1)*oldCol])
		m.matrix[i] = temp[i*m.cols : (i+1)*m.cols]
	}
	m.data = temp

	// metadata
	tempn := make([]string, m.cols)
	copy(tempn, m.attrName)
	m.attrName = tempn
	temps := make([]map[string]int, m.cols)
	copy(temps, m.str_to_enum)
	m.str_to_enum = temps
	tempe := make([]map[int]string, m.cols)
	copy(tempe, m.enum_to_str)
	m.enum_to_str = tempe
	for i := oldCol; i < m.cols; i++ {
		m.attrName[i] = fmt.Sprintf("col_%d", i)
		m.enum_to_str[i] = make(map[int]string)
		m.enum_to_str[i][ATTR_NAME] = "real"
	}
	return m
}

// Scale scales all element by the factor.
func (m *Matrix) Scale(c float64) *Matrix {
	floats.Scale(c, m.data)
	//for i := 0; i < len(m.data); i++ {
	//m.data[i] *= c
	//}
	return m
}

// ChangeAttrName changes the names of a list of attributes. If the
// new name is an empty string or is not defined, the old name is
// kept.
func (m *Matrix) ChangeAttrName(newNames []string) {
	for i := 0; i < m.cols && i < len(newNames); i++ {
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
	Require(err == nil, "SaveARFF: %v\n", err)
	defer fileio.Close()

	file := bufio.NewWriter(fileio)
	file.WriteString("@relation " + m.relation + "\n\n")
	file.Flush()
	for i := 0; i < m.cols; i++ {
		quote := ""
		for j := 0; j < len(m.attrName[i]); j++ {
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
			for j := 1; j < len(m.str_to_enum[i]); j++ {
				file.WriteString("," + m.enum_to_str[i][j])
			}
			file.WriteString("}\n")
		case 'r': // real
			file.WriteString("real\n")
		default: // date
			file.WriteString("\"yyyy-MM-dd HH:mm:ss\"\n")
		}
		file.Flush()
	}
	file.WriteString("\n@data\n")
	// write data
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if j > 0 {
				file.WriteString(",")
			}
			var s string
			switch m.enum_to_str[j][ATTR_NAME][0] {
			case 'n': // nominal
				s = fmt.Sprintf("%s", m.enum_to_str[j][int(m.matrix[i][j])])
			case 'r': // real
				s = fmt.Sprintf("%.15e", m.matrix[i][j])
			default: // date
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
func (m *Matrix) LoadARFF(fileName string, tz ...string) *Matrix {
	timeZone := "-06:00"
	if len(tz) > 0 {
		timeZone = tz[0]
	}
	fileio, err := os.Open(fileName)
	Require(err == nil, "LoadARFF: %v\n", err)
	defer fileio.Close()

	// metadata
	m.attrName = make([]string, 0, EXTRA_NUM_CELL)
	m.str_to_enum = make([]map[string]int, 0, EXTRA_NUM_CELL)
	m.enum_to_str = make([]map[int]string, 0, EXTRA_NUM_CELL)

	attrFile := bufio.NewReader(fileio)
	var line string

	// read attributes' names and data types.
	lineNum := 0 // current line
	m.cols = 0
	for err == nil {
		line, err = attrFile.ReadString('\n')
		if err != nil {
			break
		}
		lineNum++
		line = line[:len(line)-1]
		s := Split(line, "\t ", 0, 3)
		if len(s) == 0 {
			continue
		}
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
				sn := s[2][1 : len(s[2])-1]
				n := Split(sn, ",", 0)
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
				Require(s[3] == "yyyy-MM-dd HH:mm:ss" ||
					s[3] == "yyyy-MM-ddTHH:mm:ss",
					"LoadARFF: only date formats '%s' and '%s' are supported\n",
					"yyyy-MM-dd HH:mm:ss", "yyyy-MM-ddTHH:mm:ss")
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
	m.matrix = make([]Vector, 0, EXTRA_NUM_CELL)
	err = nil
	row := 0
	for err == nil {
		line, err = attrFile.ReadString('\n')
		if err != nil {
			break
		}
		lineNum++
		line = line[:len(line)-1]
		if len(line) == 0 || line[0] == '%' {
			continue
		}

		s := Split(line, "\t ,", 0)
		Require(len(s) == m.cols,
			"LoadARFF: %s: wrong number of attributes on line %d\n",
			fileName, lineNum)
		m.matrix = append(m.matrix, make([]float64, m.cols))
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
					Require(parseErr == nil || parseErr != strconv.ErrSyntax,
						"LoadARFF: %s: data at column %d on line %d is not a valid real number\n",
						fileName, j, lineNum)
					m.matrix[row][j] = number
				case 'd': // date
					t, parseErr := time.Parse(time.RFC3339,
						strings.Replace(s[j], " ", "T", 1)+timeZone)
					Require(parseErr == nil,
						"LoadARFF: %s: data at column %d on line %d is not a valid date\n",
						fileName, j, lineNum)
					m.matrix[row][j] = float64(t.Unix())
				default: // string is not implemented
				}
			}
		}
		row++
	}
	m.rows = row
	m.data = make(Vector, m.rows*m.cols)
	for i := 0; i < m.rows; i++ {
		copy(m.data[i*m.cols:], m.matrix[i])
		m.matrix[i] = m.data[i*m.cols : (i+1)*m.cols]
	}
	return m
}

// CopyRows copies rows between start[i] and end[i] from a matrix s
// to the matrix m. It will truncate any data that is out of bound on
// the matrix m.
func (m *Matrix) CopyRows(s *Matrix, start, end []int) {
	N := len(start)
	if N > len(end) {
		N = len(end)
	}

	copy(m.attrName, s.attrName)
	copy(m.str_to_enum, s.str_to_enum)
	copy(m.enum_to_str, s.enum_to_str)

	row := 0
	for j := start[0]; j < end[0] && j < m.rows && j < s.rows; j++ {
		copy(m.matrix[row], s.matrix[j])
		row++
	}
	for i := 1; i < N; i++ {
		if start[i] < end[i-1] {
			start[i] = end[i-1]
		}
		for j := start[i]; j < end[i] && j < m.rows && j < s.rows; j++ {
			copy(m.matrix[row], s.matrix[j])
			row++
		}
	}
}

// WrapRowsCols wraps a matrix around rows between start[i] and end[i]
// of matrix m for columns from colBegin to colEnd. colBegin must be less
// than s.cols and colEnd must be larger than 0.
// colBegin = colIdx[0] and colEnd = colIdx[1].
func (m *Matrix) WrapRows(s *Matrix, start, end []int, colIdx ...int) {
	colBegin := 0
	colEnd := s.cols
	if len(colIdx) > 0 {
		if colIdx[0] > 0 && colIdx[0] < s.cols {
			colBegin = colIdx[0]
		}
		if len(colIdx) > 1 {
			if colIdx[1] > colBegin && colIdx[1] <= s.cols {
				colEnd = colIdx[1]
			}
		}
	}
	m.cols = colEnd - colBegin
	if cap(m.matrix) < s.rows {
		m.matrix = make([]Vector, 0, s.rows)
	} else {
		m.matrix = m.matrix[:0]
	}

	m.relation = s.relation
	m.attrName = s.attrName[colBegin:colEnd]
	m.str_to_enum = s.str_to_enum[colBegin:colEnd]
	m.enum_to_str = s.enum_to_str[colBegin:colEnd]

	N := len(start)
	if N > len(end) {
		N = len(end)
	}

	row := 0
	for j := start[0]; j < end[0] && j < s.rows; j++ {
		m.matrix = append(m.matrix, s.matrix[j][colBegin:colEnd])
		row++
	}
	for i := 1; i < N; i++ {
		if start[i] < end[i-1] {
			start[i] = end[i-1]
		}
		for j := start[i]; j < end[i] && j < s.rows; j++ {
			m.matrix = append(m.matrix, s.matrix[j][colBegin:colEnd])
			row++
		}
	}
	m.rows = row
}

// SubMatrix copy the content of a submatrix of a matrix determined
// by the list of column indices and row indices. This operation
// allows duplicating rows and columns.
func (m *Matrix) SubMatrix(s *Matrix, rows, cols []int) {
	if len(rows) != m.rows || len(cols) != m.cols {
		*m = *NewMatrix(len(rows), len(cols), nil)
	}
	for i := 0; i < len(rows); i++ {
		for j := 0; j < len(cols); j++ {
			m.matrix[i][j] = s.matrix[rows[i]][cols[j]]
		}
	}
}

// SwapRows swaps two rows in the matrix.
func (m *Matrix) SwapRows(r1, r2 int) *Matrix {
	if r1 < 0 {
		r1 += m.rows
	}
	if r2 < 0 {
		r2 += m.rows
	}
	m.matrix[r1], m.matrix[r2] = m.matrix[r2], m.matrix[r1]
	return m
}

// SwapCols swaps two columns in the matrix (including metadata).
func (m *Matrix) SwapCols(c1, c2 int) *Matrix {
	if c1 < 0 {
		c1 += m.cols
	}
	if c2 < 0 {
		c2 += m.cols
	}
	m.attrName[c1], m.attrName[c2] = m.attrName[c2], m.attrName[c1]
	m.enum_to_str[c1], m.enum_to_str[c2] = m.enum_to_str[c2], m.enum_to_str[c1]
	m.str_to_enum[c1], m.str_to_enum[c2] = m.str_to_enum[c2], m.str_to_enum[c1]
	for i := 0; i < m.rows; i++ {
		m.matrix[i][c1], m.matrix[i][c2] = m.matrix[i][c2], m.matrix[i][c1]
	}
	return m
}

// axb returns the vector A*x + sign*b where sign is in {-1, 1}.
func (a *Matrix) axb(x, b Vector, sign int) Vector {
	Require(a.cols == len(x) && a.rows == len(b),
		"Axb: dimension mismatched: a.cols == len(x) && a.rows == len(b)\n")
	v := NewVector(a.rows, nil)
	if sign < 0 {
		for i := 0; i < len(v); i++ {
			for j := 0; j < a.cols; j++ {
				v[i] += a.matrix[i][j] * x[j]
			}
			v[i] -= b[i]
		}
	} else {
		for i := 0; i < len(v); i++ {
			for j := 0; j < a.cols; j++ {
				v[i] += a.matrix[i][j] * x[j]
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
	if m.cols != len(P) {
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
			m.SwapCols(prev, next)
			P[prev] = -1 - next
			prev = next
		}
		P[prev] = -1 - P[prev]
	}

	// recover P
	for i := 0; i < len(P); i++ {
		P[i] = -1 - P[i]
	}
}

// PermuteRows change the rows of the matrix due to a permutation.
func (m *Matrix) PermuteRows(P permutation) {
	if m.rows != len(P) {
		panic("PermuteRows: permutation is of the wrong size\n")
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
			m.SwapRows(prev, next)
			P[prev] = -1 - next
			prev = next
		}
		P[prev] = -1 - P[prev]
	}

	// recover P
	for i := 0; i < len(P); i++ {
		P[i] = -1 - P[i]
	}
}

// LeastSquare solves the solution to the least square problem with
// assumption that the norm of the solution is minimum.
// We use the pivoting Householder QR method.
// We will solve for M: min_{j}||M*x_j + b - y_j||_2 where b and m
// are column vectors. This is to support labels
// of dimension > 1.
// The matrix m in the code is the matrix formed by 1 and x_j. The
// matrix y is the labels. The output is a matrix whose first rows is
// the vector b and the remaining rows form the matrix M. Each
// columns of the output matrix is a solution.
func (m *Matrix) LeastSquare(y *Matrix) *Matrix {
	Require(m.rows >= m.cols,
		"LeastSquare: expected at least as many rows as columns\n")
	Require(m.rows == y.rows, "LeastSquare: dimension mismatched\n")

	// The solution x (the weights) is a matrix.
	x := NewMatrix(m.cols, y.cols, nil)
	var vk Vector = NewVector(m.rows, nil)

	var rP, cP permutation
	rank, vk1 := m.QR(&rP, &cP)

	// compute (Q*)b
	y.PermuteRows(rP)

	for k := 0; k < rank; k++ {
		vk[k] = vk1[k]
		l := cP[k]
		for i := k + 1; i < m.rows; i++ {
			vk[i] = m.matrix[i][l]
		}

		for t := 0; t < y.cols; t++ {
			colNorm := 2 * y.Col(t)[k:].Dot(vk[k:])
			for i := k; i < m.rows; i++ {
				y.matrix[i][t] -= colNorm * vk[i]
			}
		}
	}

	if rank < m.cols { // if rank is deficient
		oldrow := m.rows
		m.rows = rank
		n := m.Transpose()
		m.rows = oldrow

		// clear footprints of vk in n
		for i := 0; i < rank-1; i++ {
			for j := i + 1; j < rank; j++ {
				n.matrix[cP[i]][j] = 0
			}
		}

		y.matrix = y.matrix[:rank]
		vk = vk[:n.rows]
		_, vk1 := n.QR(&rP, &cP)

		// forward substitution
		for t := 0; t < y.cols; t++ {
			x.matrix[0][t] = y.matrix[cP[0]][t] / n.matrix[0][cP[0]]
			for i := 1; i < rank; i++ {
				l := cP[i]
				x.matrix[i][t] = y.matrix[l][t]
				for j := 0; j < i; j++ {
					x.matrix[i][t] -= x.matrix[j][t] * n.matrix[j][l]
				}
				x.matrix[i][t] /= n.matrix[i][l]
			}
		}

		// compute Qx
		for k := rank - 1; k >= 0; k-- {
			vk[k] = vk1[k]
			l := cP[k]
			for i := k + 1; i < n.rows; i++ {
				vk[i] = n.matrix[i][l]
			}

			for t := 0; t < y.cols; t++ {
				xDotVk := 2 * x.Col(t)[k:].Dot(vk[k:])
				for i := k; i < n.rows; i++ {
					x.matrix[i][t] -= xDotVk * vk[i]
				}
			}
		}

		for t := 0; t < y.cols; t++ {
			for i := 0; i < m.cols; i++ {
				n.matrix[i][0] = x.matrix[i][t]
			}
			for i := 0; i < m.cols; i++ {
				x.matrix[rP[i]][t] = n.matrix[i][0]
			}
		}
	} else { // full rank
		// backward substitution
		u := rank - 1
		for t := 0; t < y.cols; t++ {
			k := cP[u]
			x.matrix[k][t] = y.matrix[u][t] / m.matrix[u][k]
			var l int
			for i := rank - 2; i >= 0; i-- {
				k = cP[i]
				x.matrix[k][t] = y.matrix[i][t]
				for j := i + 1; j < rank; j++ {
					l = cP[j]
					x.matrix[k][t] -= x.matrix[l][t] * m.matrix[i][l]
				}
				x.matrix[k][t] /= m.matrix[i][k]
			}
		}
	}

	return x
}

// QR perform QR factorization on the matrix m. It stores the row and
// column pivoting in the two permutation maps rP and cP
// respectively. It returns the rank of m and a vector containing the
// first elements of vk. The remaining elements of vk's are store in
// the matrix m. The permutation maps are as follows.
// (*cP)[0] is the original column of column 0, i.e, column (*cP)[0] of
// the matrix R returned by this QR code is the column 0 of the
// matrix R after doing QR factoriation with column pivoting.
//
// NOTE: the rows of m are sorted but the columns stay still.
func (m *Matrix) QR(rP, cP *permutation) (int, Vector) {
	*rP = make([]int, m.rows)
	var P permutation = *rP

	// pre-sort rows by ascending order
	rowNorms := make(valuesort, m.rows)
	for i := 0; i < m.rows; i++ {
		rowNorms[i].val = m.Row(i).Norm(0)
		rowNorms[i].idx = i
	}
	sort.Sort(rowNorms)
	for i := 0; i < m.rows; i++ {
		P[i] = rowNorms[i].idx
	}

	// sort the rows based on sorted norms, after this P will become
	// the identity permuation.
	m.PermuteRows(P)

	// now perform Householder with column pivoting
	// We will perform column pivoting implicitly using a permutation.

	// Identity permutation on columns.
	*cP = make([]int, m.cols)
	P = *cP
	for i := 0; i < len(P); i++ {
		P[i] = i
	}

	// Householder step on both A and b.
	maxColNorm := float64(0)
	maxColIndex := 0
	colNorm := maxColNorm
	vk := NewVector(m.rows, nil)
	x := NewVector(m.rows, nil)
	vk1 := NewVector(m.cols, nil)
	var l int
	for k := 0; k < m.cols; k++ {
		// determine column with largest 2-norm
		maxColNorm = 0
		for j := k; j < m.cols; j++ {
			colNorm = 0
			l = P[j]
			for i := k; i < m.rows; i++ {
				colNorm += m.matrix[i][l] * m.matrix[i][l]
			}
			Require(!math.IsNaN(colNorm) && !math.IsInf(colNorm, 0),
				"LeastSquare: the norm of column %d is not a number.\n", P[j])
			if colNorm > maxColNorm {
				maxColNorm = colNorm
				maxColIndex = j
			}
		}

		a00 := m.matrix[0][P[0]]
		if maxColNorm < a00*ESP*ESP*a00 {
			vk1 = vk1[:k]
			return k, vk1
		}
		// swap columns i and maxColIndex
		P[k], P[maxColIndex] = P[maxColIndex], P[k]

		// Get the k-th column vector
		l = P[k]
		for j := k; j < m.rows; j++ {
			vk[j] = m.matrix[j][l]
		}

		colNorm = vk[k:].Norm(2)
		if vk[k] < 0 {
			vk[k] -= colNorm
		} else {
			vk[k] += colNorm
		}

		// A[k:m, k] -= 2vk(vk*A[k:m, k])
		if m.matrix[k][P[k]] > 0 {
			m.matrix[k][P[k]] = -colNorm
		} else {
			m.matrix[k][P[k]] = colNorm
		}

		colNorm = vk[k:].Norm(2)
		colNorm = float64(1) / colNorm
		vk[k:].Scale(colNorm)

		// save the first element to vk1
		vk1[k] = vk[k]

		// save the remaining elements of vk to m
		l = P[k]
		for i := k + 1; i < m.rows; i++ {
			m.matrix[i][l] = vk[i]
		}

		// A[k:m, k+1:n] -= 2vk(vk*A[k:m, k+1:n])
		for j := k + 1; j < m.cols; j++ {
			l = P[j]
			for i := k; i < m.rows; i++ {
				x[i] = m.matrix[i][l]
			}
			colNorm = 2 * x[k:].Dot(vk[k:])
			for i := k; i < m.rows; i++ {
				m.matrix[i][l] -= colNorm * vk[i]
			}
		}
	}
	return m.cols, vk1
}

// Random set a random normal value (0, 1) for each element of the
// matrix m.
func (m *Matrix) Random(seed ...uint64) *Matrix {
	var s uint64 = 1982
	if len(seed) > 0 {
		s = seed[0]
	}
	r := rand.NewRand(s)
	for i := 0; i < len(m.data); i++ {
		m.data[i] = r.Normal()
	}
	return m
}

// ToVector wraps a vector around m.data.
func (m *Matrix) ToVector() Vector {
	return m.data
}

// OLS return the weights in approximating labels = M*features + b.
func OLS(features, labels *Matrix) Vector {
	x := NewMatrix(features.Rows(), features.Cols()+1, nil)
	y := NewMatrix(labels.Rows(), labels.Cols(), nil)
	start := []int{0}
	end := []int{x.Rows()}
	x.CopyRows(features, start, end)
	y.CopyRows(labels, start, end)
	x.FillCol(-1, 1)
	return x.LeastSquare(y).ToVector()
}
