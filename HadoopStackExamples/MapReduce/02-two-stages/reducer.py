import sys

current_key = None
current_count = 0

for line in sys.stdin:
    key, value = line.strip().split()
    if key != current_key:
        # dump output
        if current_key is not None:
            print(f'{current_key}\t{current_count}')
        current_key = key
        current_count = 0
    
    current_count += int(value)

if current_key is not None:
    print(f'{current_key}\t{current_count}')
