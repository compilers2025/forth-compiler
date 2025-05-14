import re
import subprocess
import os


TOKENS = [
    "NUMBER",       # Numbers
    "IDENTIFIER",   # Variable or function names
    "PLUS",         # +
    "MINUS",        # -
    "MULTIPLY",     # *
    "DIVIDE",       # /
    "COLON",        # :
    "SEMICOLON",    # ;
    "IF",           # IF
    "ELSE",         # ELSE
    "THEN",         # THEN
    "LPAREN",       # (
    "RPAREN",       # )
    "DOT",          # .
    "WHITESPACE",   # Spaces, tabs, newlines
]


TOKEN_REGEX = {
    "NUMBER": r"\d+",
    "IDENTIFIER": r"[a-zA-Z_][a-zA-Z0-9_]*",
    "PLUS": r"\+",
    "MINUS": r"-",
    "MULTIPLY": r"\*",
    "DIVIDE": r"/",
    "COLON": r":",
    "SEMICOLON": r";",
    "IF": r"\bIF\b",
    "ELSE": r"\bELSE\b",
    "THEN": r"\bTHEN\b",
    "LPAREN": r"\(",
    "RPAREN": r"\)",
    "DOT": r"\.",
    "WHITESPACE": r"\s+",
}


class Token:
    def __init__(self, type, value, position):
        self.type = type
        self.value = value
        self.position = position

    def __repr__(self):
        return f"Token({self.type}, {self.value}, {self.position})"


class Lexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.position = 0

    def tokenize(self):
        tokens = []
        while self.position < len(self.source_code):
            match = None
            for token_type, regex in TOKEN_REGEX.items():
                pattern = re.compile(regex)
                match = pattern.match(self.source_code, self.position)
                if match:
                    value = match.group(0)
                    if token_type != "WHITESPACE":  # Ignore whitespace tokens
                        tokens.append(Token(token_type, value, self.position))
                    self.position += len(value)
                    break
            if not match:
                raise ValueError(
                    f"Unexpected character at position {self.position}: {self.source_code[self.position]}")
        return tokens


class ASTNode:
    pass


class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"NumberNode({self.value})"


class WordNode(ASTNode):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"WordNode({self.name})"


class DefinitionNode(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"DefinitionNode(name={self.name}, body={self.body})"


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def current_token(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def consume(self):
        token = self.current_token()
        self.position += 1
        return token

    def parse(self):
        ast = []
        while self.current_token():
            token = self.current_token()
            if token.type == "COLON":
                ast.append(self.parse_definition())
            elif token.type == "NUMBER":
                ast.append(NumberNode(int(token.value)))
                self.consume()
            elif token.type == "IDENTIFIER":
                ast.append(WordNode(token.value))
                self.consume()
            elif token.type == "DOT":
                ast.append(WordNode("."))
                self.consume()
            elif token.type in ["PLUS", "MINUS", "MULTIPLY", "DIVIDE"]:
                ast.append(WordNode(token.value))
                self.consume()
            else:
                raise ValueError(f"Unexpected token: {token}")
        return ast

    def parse_definition(self):

        self.consume()
        name_token = self.consume()
        if name_token.type != "IDENTIFIER":
            raise ValueError("Expected identifier after ':'")
        name = name_token.value
        body = []
        while self.current_token() and self.current_token().type != "SEMICOLON":
            token = self.current_token()
            if token.type == "NUMBER":
                body.append(NumberNode(int(token.value)))
                self.consume()
            elif token.type == "IDENTIFIER":
                body.append(WordNode(token.value))
                self.consume()
            elif token.type in ["PLUS", "MINUS", "MULTIPLY", "DIVIDE"]:
                body.append(WordNode(token.value))
                self.consume()
            else:
                raise ValueError(f"Unexpected token in definition: {token}")
        if not self.current_token() or self.current_token().type != "SEMICOLON":
            raise ValueError("Expected ';' at the end of definition")
        self.consume()
        return DefinitionNode(name, body)


class SemanticAnalyzer:
    def __init__(self, ast):
        self.ast = ast
        self.defined_words = set()
        self.builtin_words = {"dup", "*", "+", "-", "/",
                              ".", "drop", "swap", "square"}

    def analyze(self):
        for node in self.ast:
            if isinstance(node, DefinitionNode):
                self._check_definition(node)
            elif isinstance(node, WordNode):
                self._check_word(node.name)
            elif isinstance(node, NumberNode):
                continue
            else:
                raise ValueError(f"Unknown AST node type: {node}")

    def _check_definition(self, node):
        if node.name in self.defined_words:
            raise ValueError(f"Word '{node.name}' already defined.")
        self.defined_words.add(node.name)

        for part in node.body:
            if isinstance(part, WordNode):
                self._check_word(part.name)
            elif isinstance(part, NumberNode):
                continue
            else:
                raise ValueError(f"Invalid node in word body: {part}")

    def _check_word(self, name):
        if name not in self.defined_words and name not in self.builtin_words:
            raise ValueError(f"Undefined word used: '{name}'")


class ForthVM:
    def __init__(self):
        self.stack = []
        self.registers = {'rax': 0, 'rbx': 0}
        self.words = {}

    def run_assembly(self, assembly_file):
        try:
            with open(assembly_file, 'r') as f:
                assembly = f.readlines()

            # Process each assembly instruction
            for line in assembly:
                line = line.strip()

                # Skip empty lines, sections, and labels
                if not line or line.startswith('.') or line.endswith(':'):
                    continue

                # Handle number pushing
                if 'movq $' in line:
                    match = re.search(r'\$(\d+)', line)
                    if match:
                        value = int(match.group(1))
                        self.registers['rax'] = value

                elif 'pushq %rax' in line:
                    self.stack.append(self.registers['rax'])

                elif 'popq %rax' in line:
                    if self.stack:
                        self.registers['rax'] = self.stack.pop()

                elif 'popq %rbx' in line:
                    if self.stack:
                        self.registers['rbx'] = self.stack.pop()

                elif 'imulq %rbx, %rax' in line:
                    self.registers['rax'] *= self.registers['rbx']
                    self.stack.append(self.registers['rax'])

                elif 'imulq %rax, %rax' in line:
                    # Handle square operation
                    self.registers['rax'] *= self.registers['rax']
                    self.stack.append(self.registers['rax'])

            # Write final stack state to output.txt
            with open('output.txt', 'w') as f:
                f.write(f"Final stack: {self.stack}")

        except Exception as e:
            print(f"Error executing assembly: {str(e)}")


def generate_assembly(ast, output_file):
    assembly = [
        ".section .text",
        ".globl _start",
        "_start:"
    ]

    for node in ast:
        if isinstance(node, NumberNode):
            # Push number onto stack
            assembly.extend([
                f"    movq ${node.value}, %rax",
                "    pushq %rax"
            ])
        elif isinstance(node, WordNode):
            if node.name == "dup":
                assembly.extend([
                    "    popq %rax",
                    "    pushq %rax",
                    "    pushq %rax"
                ])
            elif node.name == "*":
                assembly.extend([
                    "    popq %rax",
                    "    popq %rbx",
                    "    imulq %rbx, %rax",
                    "    pushq %rax"
                ])
            elif node.name == "square":
                assembly.extend([
                    "    popq %rax",
                    "    imulq %rax, %rax",
                    "    pushq %rax"
                ])
        elif isinstance(node, DefinitionNode):
            # Skip definitions as they're handled through the word implementations
            continue

    # Add exit sequence
    assembly.extend([
        "    movq $60, %rax",
        "    xor %rdi, %rdi",
        "    syscall"
    ])

    with open(output_file, 'w') as f:
        f.write('\n'.join(assembly))


# Test the implementation
def main():
    try:
        # Test with the example input
        source_code = ": cube dup dup * * ; 3 cube"

        # Write to inputs.txt
        with open('inputs.txt', 'w') as f:
            f.write(source_code)

        # Read from inputs.txt
        with open('inputs.txt', 'r') as f:
            source_code = f.read()

        # Process the code
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse()

        semantic = SemanticAnalyzer(ast)
        semantic.analyze()

        # Generate and execute assembly
        generate_assembly(ast, "assembly.txt")
        print("Assembly code generated successfully.")

        vm = ForthVM()
        vm.run_assembly("assembly.txt")

        # Verify the output
        with open('output.txt', 'r') as f:
            result = f.read()
            print(f"Generated output: {result}")

    except Exception as e:
        print(f"Error in compilation process: {str(e)}")


if __name__ == "__main__":
    main()
