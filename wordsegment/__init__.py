"""English Word Segmentation in Python

Word segmentation is the process of dividing a phrase without spaces back
into its constituent parts. For example, consider a phrase like "thisisatest".
For humans, it's relatively easy to parse. This module makes it easy for
machines too. Use `segment` to parse a phrase into its parts:

>>> from wordsegment import load, segment
>>> load()
>>> segment('thisisatest')
['this', 'is', 'a', 'test']

In the code, 1024908267229 is the total number of words in the corpus. A
subset of this corpus is found in unigrams.txt and bigrams.txt which
should accompany this file. A copy of these files may be found at
http://norvig.com/ngrams/ under the names count_1w.txt and count_2w.txt
respectively.

Copyright (c) 2016 by Grant Jenks

Based on code from the chapter "Natural Language Corpus Data"
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
http://oreilly.com/catalog/9780596157111/

Original Copyright (c) 2008-2009 by Peter Norvig

"""

import io
import math
import os.path as op
import sys
from .wordsegment import Segmenter as RustSegmenter


class Segmenter(RustSegmenter):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, op.dirname(op.realpath(__file__), *args, **kwargs))


__all__ = [
    "Segmenter",
]
__title__ = "wordsegment"
__version__ = "1.3.1"
__build__ = 0x010301
__author__ = "Grant Jenks"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Grant Jenks"
