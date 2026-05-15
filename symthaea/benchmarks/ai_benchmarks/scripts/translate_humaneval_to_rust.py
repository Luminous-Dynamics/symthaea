#!/usr/bin/env python3
import json
import re
import os

TYPE_MAP = {
    "List[float]": "Vec<f64>",
    "List[int]": "Vec<i32>",
    "List[str]": "Vec<String>",
    "Dict[str, int]": "std::collections::HashMap<String, i32>",
    "Optional[int]": "Option<i32>",
    "bool": "bool",
    "float": "f64",
    "int": "i32",
    "str": "&str",
}

def translate_type(py_type):
    return TYPE_MAP.get(py_type, py_type)

def extract_metadata(prompt):
    doc_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    purpose = doc_match.group(1).strip() if doc_match else "No description"
    purpose = re.sub(r'>>>.*', '', purpose, flags=re.DOTALL).strip()
    
    # Improved signature matching
    sig_match = re.search(r'def (\w+)\s*\((.*?)\)(?:\s*->\s*(\w+))?:', prompt, re.DOTALL)
    if not sig_match:
        return None
    
    name = sig_match.group(1)
    args_raw = sig_match.group(2)
    ret_raw = sig_match.group(3) or "None"
    
    rust_args = []
    for arg in args_raw.split(','):
        if ':' in arg:
            a_name, a_type = arg.split(':')
            rust_args.append(f"{a_name.strip()}: {translate_type(a_type.strip())}")
        elif arg.strip():
            rust_args.append(f"{arg.strip()}: i32") # Default to i32
    
    signature = f"fn {name}({', '.join(rust_args)}) -> {translate_type(ret_raw)}"
    
    examples = []
    for test_match in re.finditer(r'>>> (.*?)\n\s*(.*?)\n', prompt):
        input_call = test_match.group(1).strip()
        expected = test_match.group(2).strip()
        
        # Convert Python list [1, 2] to Rust vec![1, 2] in both input and output
        input_call = re.sub(r'\[(.*?)\]', r'vec![\1]', input_call)
        
        if expected.startswith("'") or (expected.startswith('"') and not expected.endswith('")')):
            expected = f"\"{expected[1:-1]}\".to_string()"
        elif expected == "True": expected = "true"
        elif expected == "False": expected = "false"
        elif expected.startswith("["):
            expected = re.sub(r'\[(.*?)\]', r'vec![\1]', expected)
        
        examples.append({"input": input_call, "output": expected})
        
    return name, purpose, signature, examples

def main():
    input_path = "data/benchmarks/humaneval/human_eval_py.jsonl"
    output_path = "data/benchmarks/humaneval/rust_full.jsonl"
    
    count = 0
    with open(input_path, 'r') as f, open(output_path, 'w') as out:
        for line in f:
            task = json.loads(line)
            meta = extract_metadata(task['prompt'])
            if meta:
                name, purpose, signature, examples = meta
                if not examples:
                    examples = [{"input": f"{name}()", "output": "todo!()"}]
                
                rust_task = {
                    "id": task['task_id'].replace("HumanEval", "rust"),
                    "name": name,
                    "purpose": purpose,
                    "signature": signature,
                    "examples": examples
                }
                out.write(json.dumps(rust_task) + "\n")
                count += 1
                
    print(f"Successfully translated {count} tasks to {output_path}")

if __name__ == "__main__":
    main()
