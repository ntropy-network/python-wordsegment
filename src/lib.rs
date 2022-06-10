use anyhow::Context;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

static TOTAL: f64 = 1024908267229.0;

#[pyclass]
struct Segmenter {
    basepath: String,
    unigrams: HashMap<String, f64>,
    bigrams: HashMap<String, f64>,
    alphabet: HashSet<char>,
    limit: usize,
}

struct Searcher<'a> {
    unigrams: &'a HashMap<String, f64>,
    bigrams: &'a HashMap<String, f64>,
    limit: usize,
    memo: HashMap<(&'a str, &'a str), (f64, Vec<&'a str>)>,
}

impl<'a> Searcher<'a> {
    fn new(
        unigrams: &'a HashMap<String, f64>,
        bigrams: &'a HashMap<String, f64>,
        limit: usize,
    ) -> Self {
        let memo: HashMap<(&'a str, &'a str), (f64, Vec<&'a str>)> = HashMap::new();
        Searcher {
            unigrams: unigrams,
            bigrams: bigrams,
            memo: memo,
            limit: limit,
        }
    }

    fn divide(&self, text: &'a str) -> Vec<(&'a str, &'a str)> {
        // Yield `(prefix, suffix)` pairs from `text`.
        let mut res: Vec<(&str, &str)> = Vec::new();
        let end = std::cmp::min(text.len(), self.limit) + 1;
        for pos in 1..end {
            res.push(((&text[0..pos]).clone(), (&text[pos..]).clone()));
        }
        res
    }

    fn score(&mut self, word: &str, previous: Option<&str>) -> f64 {
        if previous.is_none() {
            if let Some(v) = self.unigrams.get(word) {
                return v / TOTAL;
            }

            // Penalize words not found in the unigrams according
            // to their length, a crucial heuristic.

            return 10.0 / (TOTAL * (10 ^ word.len()) as f64);
        }

        let prev = previous.unwrap();
        let bigram = format!("{} {}", prev, word);
        let bigram_res = self.bigrams.get(&bigram);
        if bigram_res.is_some() && self.unigrams.get(prev).is_some() {
            // Conditional probability of the word given the previous
            // word. The technical name is *stupid backoff* and it's
            // not a probability distribution but it works well in
            // practice.
            return bigram_res.unwrap() / TOTAL / self.score(prev, None);
        }
        // Fall back to using the unigram probability.
        return self.score(word, None);
    }

    fn search(&mut self, text: &'a str, previous: Option<&str>) -> (f64, Vec<&'a str>) {
        // Return max of candidates matching `text` given `previous` word.
        if text == "" {
            return (0.0, Vec::new());
        }

        let divided = self.divide(text);
        let mut max_candidate_value: f64 = 0.0;
        let mut max_candidate: Vec<&str> = Vec::new();
        for (prefix, suffix) in divided.into_iter() {
            let prefix_score = self.score(prefix, Some(previous.unwrap_or("<s>"))).log10();
            let pair = (suffix, prefix);
            let memo_res = self.memo.get(&pair);
            if memo_res.is_none() {
                let r = self.search(suffix, Some(prefix));
                self.memo.insert(pair, r);
            }
            let memo_res = self.memo.get(&pair);
            let (suffix_score, suffix_words) = memo_res.unwrap();
            let candidate_score = prefix_score + suffix_score;
            let mut candidate = Vec::from([prefix]);
            candidate.extend(suffix_words);
            if candidate_score > max_candidate_value {
                max_candidate_value = candidate_score;
                max_candidate = candidate;
            }
        }
        (max_candidate_value, max_candidate)
    }
}

// rust methods
impl Segmenter {
    fn py_load(&mut self, basepath: &str) -> anyhow::Result<()> {
        let path = Path::new(basepath);
        self.unigrams = self.parse(path.join("unigrams.txt").to_str().context("")?)?;
        self.bigrams = self.parse(path.join("bigrams.txt").to_str().context("")?)?;
        Ok(())
    }

    fn parse(&mut self, filename: &str) -> anyhow::Result<HashMap<String, f64>> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        let mut result: HashMap<String, f64> = HashMap::new();
        file.read_to_string(&mut contents)?;
        for line in contents.split("\n") {
            if line == "" {
                continue;
            }
            let words = line.split("\t").collect::<Vec<&str>>();
            assert_eq!(words.len(), 2);
            let word = words[0];
            let score = words[1].parse::<f64>()?;
            result.insert(word.to_string(), score);
        }
        Ok(result)
    }

    fn clean(&self, text: String) -> String {
        // Return `text` lower-cased with non-alphanumeric characters removed.
        let mut text_lower = text.to_lowercase();
        text_lower.retain(|b: char| self.alphabet.contains(&b));
        text_lower
    }

    fn do_segment<'a>(&self, text: String) -> Vec<String> {
        let mut output: Vec<String> = Vec::new();

        let mut s = Searcher::new(&self.unigrams, &self.bigrams, self.limit);

        let clean_text = self.clean(text);
        let size = 250;
        let mut prefix = "".to_string();

        let mut search_prefixes: Vec<String> = Vec::new();

        for offset in (0..clean_text.len()).step_by(size) {
            let max_ = std::cmp::min(clean_text.len(), offset + size);
            let chunk: &str = &clean_text.as_str()[offset..max_];
            let se: String = format!("{}{}", prefix.clone(), chunk.clone());
            search_prefixes.push(se);
        }

        for search_prefix in &search_prefixes {
            let (_, chunk_words) = s.search(search_prefix.as_str(), None);
            let len = chunk_words.len();
            let v = chunk_words[len - 5..len].join("");
            prefix = v;
            for word in &chunk_words[..len - 5] {
                output.push(word.to_string());
            }
        }
        let (_, prefix_words) = s.search(prefix.as_str(), None);

        for word in prefix_words.into_iter() {
            output.push(word.to_string());
        }
        output
    }
}

#[pymethods]
impl Segmenter {
    #[new]
    #[args(limit = "16")]
    fn new(basepath: &str, limit: usize) -> Self {
        Segmenter {
            basepath: basepath.to_string(),
            unigrams: HashMap::new(),
            bigrams: HashMap::new(),
            alphabet: [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5',
                '6', '7', '8', '9',
            ]
            .into_iter()
            .map(|b| b)
            .collect(),
            limit: limit,
        }
    }

    fn load(&mut self) -> PyResult<()> {
        self.py_load(&self.basepath.clone())?;
        Ok(())
    }

    fn segment(&mut self, word: String) -> PyResult<Vec<PyObject>> {
        let res = self.do_segment(word);
        let gil = Python::acquire_gil();
        let py = gil.python();

        Ok(res.into_iter().map(|a| a.to_object(py)).collect())
    }
}
/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "wordsegment")]
fn python_wordsegment(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Segmenter>()?;
    Ok(())
}
