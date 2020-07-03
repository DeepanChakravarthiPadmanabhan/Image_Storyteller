import random
num_lines = 30000
with open('/scratch/smuthi2s/NLP_data/books/books_large_p1.txt', 'r') as in_file:
# with open('/home/smuthi2s/perl5/NLP/Image_Storyteller/tf2-skip-thoughts/data.txt', 'r') as in_file:
    total_sentences = in_file.read().splitlines()[:num_lines]
print(total_sentences[random.randint(0,num_lines)])
