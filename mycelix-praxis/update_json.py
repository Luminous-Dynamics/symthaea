import sys

with open('nodes.txt', 'r') as f:
    nodes_content = f.read().strip()

with open('edges.txt', 'r') as f:
    edges_content = f.read().strip()

target_path = '/srv/luminous-dynamics/mycelix-praxis/examples/curriculum/unified_k_to_phd.json'
with open(target_path, 'r') as f:
    lines = f.readlines()

# Nodes insertion
# The user said: "Append the 10 node objects to the end of the nodes array (before the ], at line 45494)."
# Line 45494 is index 45493.
target_node_end_index = 45493 
print(f"Line {target_node_end_index + 1}: {repr(lines[target_node_end_index])}")

if '],' in lines[target_node_end_index]:
    # Previous line (index 45492) should be the last node object closing brace.
    print(f"Line {target_node_end_index}: {repr(lines[target_node_end_index-1])}")
    lines[target_node_end_index-1] = lines[target_node_end_index-1].replace('    }', '    },')
    lines.insert(target_node_end_index, nodes_content + '\n')
else:
    print("Error: Could not find node array end at expected line.")
    sys.exit(1)

# Edges insertion
# Before final ] at the end of the file.
# The file ends with:
#     ]
#   }
found_edges_end = False
for i in range(len(lines)-1, -1, -1):
    if lines[i].strip() == ']':
        print(f"Found end of edges at index {i}: {repr(lines[i])}")
        # Add comma to previous line if not present
        if not lines[i-1].strip().endswith(','):
             lines[i-1] = lines[i-1].rstrip() + ',\n'
        lines.insert(i, edges_content + '\n')
        found_edges_end = True
        break

if not found_edges_end:
    print("Error: Could not find edges array end.")
    sys.exit(1)

with open(target_path, 'w') as f:
    f.writelines(lines)
