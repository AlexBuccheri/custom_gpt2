# Generating a Custom GPT2 Model from Scratch

## Introduction

Following this [reference](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=638s)


## Tokenisers

Long sequence of integers result in very short vocabularies, whereas short sequence of integers result in very large 
vocabularies. For example, `tiktoken` will return fewer integers to tokenise "Hello, world" than a character-level 
tokeniser used [here](gpt2_from_scratch.py), but consequently the phase space of tokens massively increases 
(i.e. a large code book).