import sys
import re

pattern = re.compile(r'\W+')

stop_words = set()
with open('stop_words_en.txt', 'r') as fp:
    for line in fp:
        stop_words.add(line.strip())


for line in sys.stdin:
    article_id, text = line.strip().split('\t', 1)

    text = pattern.sub(' ', text).lower()
    
    tokens = text.split()
    for token in tokens:
        if token in stop_words or len(token) < 3:
            print(f"reporter:counter:Wiki stats,Stop words,{1}", file=sys.stderr)
            continue
        
        print(f"reporter:counter:Wiki stats,Not stop words,{1}", file=sys.stderr)
        print(f'{token}\t{1}')
