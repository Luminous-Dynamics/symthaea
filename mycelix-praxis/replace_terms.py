import os
import re

replacements = [
    # (Pattern parts, Replacement parts)
    # 1. "Consciousness Gating" -> "Epistemic Gating"
    (("Consciousness", "Gating"), ("Epistemic", "Gating")),
    # 2. "Consciousness Computing" -> "Cognitive Computing"
    (("Consciousness", "Computing"), ("Cognitive", "Computing")),
    # 3. "Consciousness level" -> "Cognitive readiness"
    (("Consciousness", "level"), ("Cognitive", "readiness")),
    # 4. "Consciousness state" -> "Cognitive state"
    (("Consciousness", "state"), ("Cognitive", "state")),
    # 5. "Consciousness provider" -> "Cognitive provider"
    (("Consciousness", "provider"), ("Cognitive", "provider")),
    # 6. "Consciousness", "aware" -> "Cognitive", "aware"
    (("Consciousness", "aware"), ("Cognitive", "aware")),
]

def get_variants(w1, w2):
    # Returns a list of (old_string, new_string)
    variants = []
    
    # Space separated (Title Case)
    variants.append((f"{w1} {w2}", f"{repl_map[(w1, w2)][0]} {repl_map[(w1, w2)][1]}"))
    # lowercase
    variants.append((f"{w1.lower()} {w2.lower()}", f"{repl_map[(w1, w2)][0].lower()} {repl_map[(w1, w2)][1].lower()}"))
    # UPPERCASE
    variants.append((f"{w1.upper()} {w2.upper()}", f"{repl_map[(w1, w2)][0].upper()} {repl_map[(w1, w2)][1].upper()}"))
    
    # snake_case
    variants.append((f"{w1.lower()}_{w2.lower()}", f"{repl_map[(w1, w2)][0].lower()}_{repl_map[(w1, w2)][1].lower()}"))
    # kebab-case
    variants.append((f"{w1.lower()}-{w2.lower()}", f"{repl_map[(w1, w2)][0].lower()}-{repl_map[(w1, w2)][1].lower()}"))
    # CamelCase
    variants.append((f"{w1}{w2}", f"{repl_map[(w1, w2)][0]}{repl_map[(w1, w2)][1]}"))
    
    # Specific case for "Consciousness-aware" (hyphenated in original)
    if w2 == "aware":
         variants.append((f"{w1}-aware", f"{repl_map[(w1, w2)][0]}-aware"))
         variants.append((f"{w1.lower()}-aware", f"{repl_map[(w1, w2)][0].lower()}-aware"))
         variants.append((f"{w1.upper()}-AWARE", f"{repl_map[(w1, w2)][0].upper()}-AWARE"))

    return variants

repl_map = {k: v for k, v in replacements}
all_variants = []
for (w1, w2) in repl_map:
    all_variants.extend(get_variants(w1, w2))

# Remove duplicates
all_variants = sorted(list(set(all_variants)), key=lambda x: len(x[0]), reverse=True)

extensions = {'.rs', '.json', '.md', '.js', '.css', '.toml', '.yaml', '.yml', '.html'}

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return # Skip binary files

    original_content = content
    for old, new in all_variants:
        content = content.replace(old, new)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {filepath}")

for root, dirs, files in os.walk('.'):
    # Skip some directories
    if '.git' in dirs:
        dirs.remove('.git')
    if 'target' in dirs:
        dirs.remove('target')
    if 'node_modules' in dirs:
        dirs.remove('node_modules')
    if '.cargo' in dirs:
        dirs.remove('.cargo')

    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            process_file(os.path.join(root, file))
