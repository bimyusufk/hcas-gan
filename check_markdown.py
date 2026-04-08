#!/usr/bin/env python3
import re
import sys

file_path = r'c:\Users\bimyu\Documents\Projects\HCAS-GAN-project\HCAS-GAN_Dokumentasi_Eksperimen_Komprehensif.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

print("=== Markdown Validation Report ===\n")

# Count backticks
backtick_count = content.count('`')
print(f'Total backticks: {backtick_count}')
if backtick_count % 2 != 0:
    print('  ⚠ WARNING: Odd number of backticks')

# Check code blocks
code_block_count = 0
in_code_block = False
code_block_lines = []
for i, line in enumerate(lines, 1):
    if line.startswith('```'):
        code_block_count += 1
        in_code_block = not in_code_block
        code_block_lines.append((i, line[:30]))

print(f'\nCode blocks (triple backticks): {code_block_count}')
if code_block_count % 2 != 0:
    print(f'  ❌ ERROR: Unclosed code block (odd number)')
    if code_block_lines:
        print("  Last code block markers:")
        for line_num, text in code_block_lines[-3:]:
            print(f"    Line {line_num}: {text}")

# Check mermaid blocks
mermaid_count = sum(1 for line in lines if '```mermaid' in line)
print(f'\nMermaid blocks: {mermaid_count}')

# Check bracket/parenthesis matching
brackets = {'[': ']', '{': '}', '(': ')'}
for btype, bclose in brackets.items():
    count_open = content.count(btype)
    count_close = content.count(bclose)
    if count_open != count_close:
        print(f'  ⚠ WARNING: Mismatched {btype}/{bclose} (open:{count_open}, close:{count_close})')

# Check for common markdown issues
print('\n=== Specific Issues ===')

# Check for inline code with missing close
inline_code_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    backtick_counter = 0
    current_pos = 0
    while current_pos < len(line):
        if line[current_pos] == '`' and (current_pos == 0 or line[current_pos-1] != '\\'):
            backtick_counter += 1
        current_pos += 1
    
    if backtick_counter > 0 and backtick_counter % 2 != 0:
        inline_code_lines.append((i+1, line[:60]))
    i += 1

if inline_code_lines:
    print(f'Lines with odd backtick count (may have unclosed inline code):')
    for line_num, text in inline_code_lines[:5]:
        print(f'  Line {line_num}: {text}...')

# Check for math mode issues  
dollar_count = content.count('$')
print(f'\nDollar signs (for math): {dollar_count}')
if dollar_count % 2 != 0:
    print('  ⚠ WARNING: Unclosed math expression')

# List all section headers
print('\n=== Section Headers ===')
section_count = 0
for i, line in enumerate(lines, 1):
    if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
        level = len(line) - len(line.lstrip('#'))
        section_count += 1
        if section_count <= 10:  # Show first 10
            print(f"  L{i}: {'  ' * (level-1)}{line[:70]}")

print(f'\nTotal sections: {section_count}')

print('\n=== Summary ===')
print(f'Total lines: {len(lines)}')
print(f'File size: {len(content)} bytes')
print('\n✓ Check complete!')
