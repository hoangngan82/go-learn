package matrix

import (
  "fmt"
  "runtime"
)

func Require(e bool, msg string, a ...interface{}) {
  if !e {
    _, fileName, line, ok := runtime.Caller(1)
    var s string
    if ok {
      s = fmt.Sprintf("At: %s:%d\n", fileName, line)
    }
    s = fmt.Sprintf(msg, a...) + s
    panic(s)
  }
}

