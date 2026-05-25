import json
import sys

def main():
    with open('Phishing_Detection_LLM_v4.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        src = "".join(cell.get('source', []))
        print(src[:200] + ("..." if len(src) > 200 else ""))
        print()

if __name__ == '__main__':
    main()
