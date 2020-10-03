# 0x10. Natural Language Processing - Evaluation Metrics

## Learning Objectives

- What are the applications of natural language processing?
- What is a BLEU score?
- What is a ROUGE score?
- What is perplexity?
- When should you use one evaluation metric over another?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- You are not allowed to use the `nltk` module

## Tasks

### [0. Unigram BLEU score](./0-uni_bleu.py)

Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:

- `references` is a list of reference translations
  - each reference translation is a list of the words in the translation
- `sentence` is a list containing the model proposed sentence
- Returns: the unigram BLEU score

```sh
$ ./0-main.py
0.6549846024623855
$
```

---

### [1. N-gram BLEU score](./1-ngram_bleu.py)

Write the function `def ngram_bleu(references, sentence, n):` that calculates the n-gram BLEU score for a sentence:

- `references` is a list of reference translations
  - each reference translation is a list of the words in the translation
- `sentence` is a list containing the model proposed sentence
- `n` is the size of the n-gram to use for evaluation
- Returns: the n-gram BLEU score

```sh
$ ./1-main.py
0.6140480648084865
$
```

---

### [2. Cumulative N-gram BLEU score](./2-cumulative_bleu.py)

Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n-gram BLEU score for a sentence:

- `references` is a list of reference translations
  - each reference translation is a list of the words in the translation
- `sentence` is a list containing the model proposed sentence
- `n` is the size of the largest n-gram to use for evaluation
- All n-gram scores should be weighted evenly
- Returns: the cumulative n-gram BLEU score

```sh
$ ./2-main.py
0.5475182535069453
$
```

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)