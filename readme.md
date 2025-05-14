# Forth Compiler in Python

This project is a simple Forth compiler and virtual machine written in Python. It supports basic arithmetic, stack manipulation, and user-defined words. The compiler parses Forth-like source code, generates simulated assembly instructions, and executes them using a custom VM.

---
# Features

-  Tokenizer for Forth syntax
-  Parser and AST generation
-  Semantic analysis (undefined word detection)
-  Assembly-like code generation
-  Stack-based virtual machine execution
-  Built-in words:
  - Arithmetic: `+`, `-`, `*`, `/`
  - Stack ops: `dup`, `drop`, `swap`
  - Control: `.`, `square`
-  Custom word definitions using `:` and `;`

# Example Inputs:

```forth 
: square dup * ; 7 2 - square
or 
: square dup * ; 7 2 - square.

# Expected output:
FINAL STACK:[25]
Empty stack
