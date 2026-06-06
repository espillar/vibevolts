import ast
import os
import glob

def get_signature(node):
    try:
        args = []
        # Positional-only arguments
        if hasattr(node.args, 'posonlyargs'):
            for arg in node.args.posonlyargs:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            if node.args.posonlyargs:
                args.append("/")

        # Standard arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
            
        # Keyword-only arguments
        if node.args.kwonlyargs:
            if not node.args.vararg:
                args.append("*")
            for arg in node.args.kwonlyargs:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
            
        sig = f"({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"
        return sig
    except Exception as e:
        return f"(Error extracting signature: {e})"

def extract_info(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            return f"## {os.path.basename(filepath)}\nError parsing file: {e}\n---\nDependencies: []"
            
    filename = os.path.basename(filepath)
    output = [f"## {filename}"]
    
    dependencies = set()
    functions = []
    
    # We want to maintain order of appearance for functions
    # ast.walk doesn't guarantee order, so we'll use ast.iter_child_nodes or just filter
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            else:
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            signature = get_signature(node)
            docstring = ast.get_docstring(node) or "No docstring."
            functions.append((name, signature, docstring))
            
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{node.name}.{item.name}"
                    signature = get_signature(item)
                    docstring = ast.get_docstring(item) or "No docstring."
                    functions.append((name, signature, docstring))

    for name, sig, doc in functions:
        output.append(f"### {name}{sig}")
        output.append(doc)
    
    output.append("---")
    # Filter out empty or None dependencies
    deps = sorted(list(filter(None, dependencies)))
    output.append(f"Dependencies: [{', '.join(deps)}]")
    return "\n".join(output)

if __name__ == "__main__":
    files = sorted(glob.glob("*.py"))
    full_output = []
    for f in files:
        full_output.append(extract_info(f))
    
    print("\n\n".join(full_output))
