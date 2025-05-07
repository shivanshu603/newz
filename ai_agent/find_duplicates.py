import re

def find_duplicate_methods(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all method declarations
    methods = re.findall(r'def\s+(\w+)', content)
    
    # Find duplicates
    duplicates = [m for m in set(methods) if methods.count(m) > 1]
    
    print(f"Duplicate methods in {filename}:")
    for method in duplicates:
        print(f"- {method}")
        # Find all occurrences of this method
        matches = re.finditer(r'def\s+' + method + r'\s*\(', content)
        for i, match in enumerate(matches):
            line_no = content[:match.start()].count('\n') + 1
            print(f"  Occurrence {i+1}: Line {line_no}")

if __name__ == "__main__":
    find_duplicate_methods('blog_generator_new.py') 