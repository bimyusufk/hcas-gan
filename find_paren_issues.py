#!/usr/bin/env python3
import re

file_path = r'c:\Users\bimyu\Documents\Projects\HCAS-GAN-project\HCAS-GAN_Dokumentasi_Eksperimen_Komprehensif.md'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== Finding Mismatched Parentheses ===\n")

# Track parenthesis balance
stack = []
issues = []

for line_num, line in enumerate(lines, 1):
    content = line.rstrip()
    
    # Skip code blocks
    if line.strip().startswith('```'):
        continue
    
    for char_pos, char in enumerate(line):
        if char == '(':
            stack.append((line_num, char_pos, '('))
        elif char == ')':
            if stack and stack[-1][2] == '(':
                stack.pop()
            else:
                issues.append((line_num, char_pos, 'Extra closing ) without matching ('))

if stack:
    print("Unclosed opening parentheses:")
    for line_num, char_pos, char in stack:
        print(f"  Line {line_num}, Pos {char_pos}: {repr(char)}")
    print()

if issues:
    print("Extra closing parentheses:")
    for line_num, char_pos, msg in issues:
        print(f"  Line {line_num}, Pos {char_pos}: {msg}")
    print()

print(f"\nSummary:")
print(f"  Unclosed opening: {len(stack)}")
print(f"  Extra closing: {len(issues)}")

# Now find those lines
if issues:
    print("\n=== Lines with issues ===")
    line_set = set(i[0] for i in issues)
    for line_num in sorted(line_set):
        print(f"\nLine {line_num}:")
        print(f"  {lines[line_num-1].rstrip()}")
