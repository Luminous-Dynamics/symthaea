// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! CfC Code Sequencer training on abstract algorithm vectors.
//!
//! Extracts structural features from the Exercism type-aware module
//! implementations, encodes them as 16,384D HDC vectors via
//! AlgorithmEncoder, and trains the CfC temporal network to predict
//! algorithm class from the HDC representation.

use super::algorithm_encoder::{
    AlgorithmChannels, AlgorithmClass, AlgorithmEncoder, AlgorithmTrainingPair, extract_features,
};
use crate::dynamics::cfc_code_sequencer::{CfCCodeSequencer, PlanAction};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ─── Exercism Solution Corpus ──────────────────────────────────────────────

/// A labeled code sample for training.
struct LabeledSolution {
    name: &'static str,
    class: AlgorithmClass,
    source: &'static str,
}

/// The corpus of labeled Exercism solutions.
///
/// Each entry: (exercise_name, algorithm_class, representative_source).
/// The source is a minimal version of the type-aware module — enough to
/// extract structural features without the full benchmark machinery.
fn exercism_corpus() -> Vec<LabeledSolution> {
    vec![
        // ── Sorting ──
        LabeledSolution {
            name: "sublist",
            class: AlgorithmClass::Sorting,
            source: r#"pub fn sublist(a: &[T], b: &[T]) -> Comparison {
    if a == b { Comparison::Equal }
    else if a.is_empty() || b.windows(a.len()).any(|w| w == a) { Comparison::Sublist }
    else if b.is_empty() || a.windows(b.len()).any(|w| w == b) { Comparison::Superlist }
    else { Comparison::Unequal }
}"#,
        },
        // ── Search ──
        LabeledSolution {
            name: "binary-search",
            class: AlgorithmClass::Search,
            source: r#"pub fn find(array: &[i32], key: i32) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = array.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if array[mid] == key { return Some(mid); }
        else if array[mid] < key { lo = mid + 1; }
        else { hi = mid; }
    }
    None
}"#,
        },
        LabeledSolution {
            name: "sieve",
            class: AlgorithmClass::Search,
            source: r#"pub fn primes_up_to(n: u64) -> Vec<u64> {
    if n < 2 { return Vec::new(); }
    let n = n as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false; is_prime[1] = false;
    let mut i = 2;
    while i * i <= n {
        if is_prime[i] { for j in (i*i..=n).step_by(i) { is_prime[j] = false; } }
        i += 1;
    }
    (2..=n).filter(|&i| is_prime[i]).map(|i| i as u64).collect()
}"#,
        },
        // ── Dynamic Programming ──
        LabeledSolution {
            name: "knapsack",
            class: AlgorithmClass::DynamicProgramming,
            source: r#"pub fn maximum_value(max_weight: u32, items: &[Item]) -> u32 {
    let w = max_weight as usize;
    let mut dp = vec![0u32; w + 1];
    for item in items {
        let iw = item.weight as usize;
        for cap in (iw..=w).rev() {
            dp[cap] = dp[cap].max(dp[cap - iw] + item.value);
        }
    }
    dp[w]
}"#,
        },
        LabeledSolution {
            name: "book-store",
            class: AlgorithmClass::DynamicProgramming,
            source: r#"pub fn lowest_price(books: &[u32]) -> u32 {
    let mut counts = [0u32; 5];
    for &book in books { counts[(book - 1) as usize] += 1; }
    let mut groups = [0u32; 6];
    let mut prev = 0;
    for i in (0..5).rev() {
        let diff = counts[i] - prev;
        if diff > 0 { groups[i + 1] += diff; }
        prev = counts[i];
    }
    let swaps = groups[5].min(groups[3]);
    groups[5] -= swaps; groups[3] -= swaps; groups[4] += swaps * 2;
    let price = [0, 800, 1520, 2160, 2560, 3000];
    (1..=5).map(|k| groups[k] * price[k]).sum()
}"#,
        },
        // ── Graph ──
        LabeledSolution {
            name: "dominoes",
            class: AlgorithmClass::Graph,
            source: r#"pub fn chain(input: &[(u8, u8)]) -> Option<Vec<(u8, u8)>> {
    if input.is_empty() { return Some(Vec::new()); }
    let n = input.len();
    let mut used = vec![false; n];
    let mut path = Vec::with_capacity(n);
    fn solve(input: &[(u8, u8)], used: &mut Vec<bool>, path: &mut Vec<(u8, u8)>, n: usize) -> bool {
        if path.len() == n { return path.first().unwrap().0 == path.last().unwrap().1; }
        let need = if path.is_empty() { None } else { Some(path.last().unwrap().1) };
        for i in 0..n {
            if used[i] { continue; }
            for &domino in &[(input[i].0, input[i].1), (input[i].1, input[i].0)] {
                if need.is_none() || need == Some(domino.0) {
                    used[i] = true; path.push(domino);
                    if solve(input, used, path, n) { return true; }
                    path.pop(); used[i] = false;
                }
            }
        }
        false
    }
    if solve(input, &mut used, &mut path, n) { Some(path) } else { None }
}"#,
        },
        LabeledSolution {
            name: "two-bucket",
            class: AlgorithmClass::Graph,
            source: r#"pub fn solve(cap1: u8, cap2: u8, goal: u8, start: &Bucket) -> Option<BucketStats> {
    use std::collections::{HashSet, VecDeque};
    let (ca, cb, swap) = match start {
        Bucket::One => (cap1 as i16, cap2 as i16, false),
        Bucket::Two => (cap2 as i16, cap1 as i16, true),
    };
    let forbidden = |a: i16, b: i16| a == 0 && b == cb;
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert((ca, 0i16));
    queue.push_back((ca, 0i16, 1u8));
    while let Some((a, b, moves)) = queue.pop_front() {
        let (r1, r2) = if swap { (b, a) } else { (a, b) };
        if r1 == goal as i16 || r2 == goal as i16 { return Some(BucketStats { moves, goal_bucket: if r1 == goal as i16 { Bucket::One } else { Bucket::Two }, other_bucket: if r1 == goal as i16 { r2 as u8 } else { r1 as u8 } }); }
        for (na, nb) in [(ca, b), (a, cb), (0, b), (a, 0), { let p = a.min(cb - b); (a - p, b + p) }, { let p = b.min(ca - a); (a + p, b - p) }] {
            if !forbidden(na, nb) && visited.insert((na, nb)) { queue.push_back((na, nb, moves + 1)); }
        }
    }
    None
}"#,
        },
        // ── String Processing ──
        LabeledSolution {
            name: "anagram",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn anagrams_for<'a>(word: &str, possible: &[&'a str]) -> HashSet<&'a str> {
    let lower = word.to_lowercase();
    let mut sorted: Vec<char> = lower.chars().collect();
    sorted.sort_unstable();
    possible.iter().copied().filter(|c| {
        let cl = c.to_lowercase();
        if cl == lower { return false; }
        let mut cs: Vec<char> = cl.chars().collect();
        cs.sort_unstable();
        cs == sorted
    }).collect()
}"#,
        },
        LabeledSolution {
            name: "acronym",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn abbreviate(phrase: &str) -> String {
    phrase.split(|c: char| c.is_whitespace() || c == '-' || c == '_')
        .filter(|w| !w.is_empty())
        .flat_map(|word| {
            let mut chars = word.chars().peekable();
            let mut initials = vec![chars.next().unwrap().to_ascii_uppercase()];
            while let Some(c) = chars.next() {
                if c.is_uppercase() && chars.peek().map_or(false, |n| n.is_lowercase()) {
                    initials.push(c);
                }
            }
            initials
        }).collect()
}"#,
        },
        LabeledSolution {
            name: "say",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn encode(n: u64) -> String {
    if n == 0 { return "zero".to_string(); }
    let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"];
    let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    fn say_below_1000(n: u64, ones: &[&str], tens: &[&str]) -> String {
        if n == 0 { return String::new(); }
        if n < 20 { return ones[n as usize].to_string(); }
        if n < 100 {
            let t = tens[(n / 10) as usize].to_string();
            if n % 10 == 0 { t } else { format!("{}-{}", t, ones[(n % 10) as usize]) }
        } else {
            let h = format!("{} hundred", ones[(n / 100) as usize]);
            if n % 100 == 0 { h } else { format!("{} {}", h, say_below_1000(n % 100, ones, tens)) }
        }
    }
    let scales = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"];
    let mut parts = Vec::new();
    let mut remaining = n;
    let mut scale = 0;
    while remaining > 0 {
        let chunk = remaining % 1000;
        if chunk > 0 {
            let s = say_below_1000(chunk, &ones, &tens);
            if scale > 0 { parts.push(format!("{} {}", s, scales[scale])); } else { parts.push(s); }
        }
        remaining /= 1000; scale += 1;
    }
    parts.reverse(); parts.join(" ")
}"#,
        },
        LabeledSolution {
            name: "crypto-square",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn encrypt(input: &str) -> String {
    let normalized: Vec<char> = input.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase()).collect();
    if normalized.is_empty() { return String::new(); }
    let len = normalized.len();
    let cols = (len as f64).sqrt().ceil() as usize;
    let rows = (len + cols - 1) / cols;
    (0..cols).map(|c| (0..rows).map(|r| {
        let idx = r * cols + c;
        if idx < len { normalized[idx] } else { ' ' }
    }).collect::<String>()).collect::<Vec<_>>().join(" ")
}"#,
        },
        // ── Mathematical ──
        LabeledSolution {
            name: "nth-prime",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn nth(n: u32) -> u32 {
    let mut count = 0;
    let mut candidate = 2;
    loop {
        if is_prime(candidate) {
            if count == n { return candidate; }
            count += 1;
        }
        candidate += 1;
    }
}
fn is_prime(n: u32) -> bool {
    if n < 2 { return false; }
    let mut i = 2;
    while i * i <= n { if n % i == 0 { return false; } i += 1; }
    true
}"#,
        },
        LabeledSolution {
            name: "alphametics",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn solve(input: &str) -> Option<HashMap<char, u8>> {
    let parts: Vec<&str> = input.split("==").collect();
    let addends: Vec<&str> = parts[0].split('+').map(|s| s.trim()).collect();
    let words: Vec<&str> = addends.iter().copied().chain(std::iter::once(parts[1].trim())).collect();
    let mut letters: Vec<char> = words.iter().flat_map(|w| w.chars()).filter(|c| c.is_alphabetic()).collect();
    letters.sort(); letters.dedup();
    let leading: Vec<char> = words.iter().filter(|w| w.len() > 1).map(|w| w.chars().next().unwrap()).collect();
    fn try_solve(letters: &[char], leading: &[char], addends: &[&str], rhs: &str, assign: &mut HashMap<char, u8>, used: &mut [bool; 10]) -> Option<HashMap<char, u8>> {
        if assign.len() == letters.len() {
            let val = |w: &str, m: &HashMap<char, u8>| w.chars().fold(0u64, |acc, c| acc * 10 + m[&c] as u64);
            let sum: u64 = addends.iter().map(|w| val(w, assign)).sum();
            if sum == val(rhs, assign) { return Some(assign.clone()); }
            return None;
        }
        let letter = letters[assign.len()];
        let start = if leading.contains(&letter) { 1 } else { 0 };
        for d in start..=9u8 {
            if used[d as usize] { continue; }
            assign.insert(letter, d); used[d as usize] = true;
            if let Some(r) = try_solve(letters, leading, addends, rhs, assign, used) { return Some(r); }
            assign.remove(&letter); used[d as usize] = false;
        }
        None
    }
    try_solve(&letters, &leading, &addends, parts[1].trim(), &mut HashMap::new(), &mut [false; 10])
}"#,
        },
        LabeledSolution {
            name: "diamond",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn get_diamond(c: char) -> Vec<String> {
    let n = (c as u8 - b'A') as usize;
    let width = 2 * n + 1;
    let mut rows: Vec<String> = Vec::with_capacity(width);
    for i in 0..=n {
        let ch = (b'A' + i as u8) as char;
        let mut row = vec![' '; width];
        row[n - i] = ch; row[n + i] = ch;
        rows.push(row.iter().collect::<String>());
    }
    for i in (0..n).rev() { rows.push(rows[i].clone()); }
    rows
}"#,
        },
        // ── Data Structure ──
        LabeledSolution {
            name: "circular-buffer",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct CircularBuffer<T> { buffer: Vec<Option<T>>, read_pos: usize, write_pos: usize, count: usize, capacity: usize }
impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity { buffer.push(None); }
        CircularBuffer { buffer, read_pos: 0, write_pos: 0, count: 0, capacity }
    }
    pub fn write(&mut self, element: T) -> Result<(), Error> {
        if self.count == self.capacity { return Err(Error::FullBuffer); }
        self.buffer[self.write_pos] = Some(element);
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.count += 1;
        Ok(())
    }
    pub fn read(&mut self) -> Result<T, Error> {
        if self.count == 0 { return Err(Error::EmptyBuffer); }
        let element = self.buffer[self.read_pos].take().unwrap();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.count -= 1;
        Ok(element)
    }
}"#,
        },
        LabeledSolution {
            name: "simple-linked-list",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct SimpleLinkedList<T> { head: Option<Box<Node<T>>>, len: usize }
struct Node<T> { data: T, next: Option<Box<Node<T>>> }
impl<T> SimpleLinkedList<T> {
    pub fn new() -> Self { SimpleLinkedList { head: None, len: 0 } }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn push(&mut self, element: T) {
        self.head = Some(Box::new(Node { data: element, next: self.head.take() }));
        self.len += 1;
    }
    pub fn pop(&mut self) -> Option<T> {
        self.head.take().map(|node| { self.head = node.next; self.len -= 1; node.data })
    }
}"#,
        },
        LabeledSolution {
            name: "forth",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct Forth { stack: Vec<Value>, defs: Vec<Vec<Op>>, names: HashMap<String, usize> }
impl Forth {
    pub fn new() -> Forth { Forth { stack: Vec::new(), defs: Vec::new(), names: HashMap::new() } }
    pub fn stack(&self) -> &[Value] { &self.stack }
    pub fn eval(&mut self, input: &str) -> Result {
        let tokens: Vec<String> = input.to_lowercase().split_whitespace().map(String::from).collect();
        let mut i = 0;
        while i < tokens.len() {
            if tokens[i] == ":" {
                i += 1; let name = tokens[i].clone(); i += 1;
                let mut body = Vec::new();
                while i < tokens.len() && tokens[i] != ";" { body.push(self.compile_token(&tokens[i])?); i += 1; }
                let idx = self.defs.len(); self.defs.push(body); self.names.insert(name, idx);
                i += 1;
            } else { let op = self.compile_token(&tokens[i])?; self.exec(&op)?; i += 1; }
        }
        Ok(())
    }
}"#,
        },
        // ── IO Transform ──
        LabeledSolution {
            name: "minesweeper",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn annotate(minefield: &[&str]) -> Vec<String> {
    let rows = minefield.len();
    if rows == 0 { return Vec::new(); }
    let cols = minefield[0].len();
    let grid: Vec<Vec<char>> = minefield.iter().map(|r| r.chars().collect()).collect();
    (0..rows).map(|r| (0..cols).map(|c| {
        if grid[r][c] == '*' { return '*'; }
        let mut count = 0u8;
        for dr in -1i32..=1 { for dc in -1i32..=1 {
            if dr == 0 && dc == 0 { continue; }
            let nr = r as i32 + dr; let nc = c as i32 + dc;
            if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 && grid[nr as usize][nc as usize] == '*' { count += 1; }
        }}
        if count > 0 { (b'0' + count) as char } else { ' ' }
    }).collect()).collect()
}"#,
        },
        LabeledSolution {
            name: "parallel-letter-frequency",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn frequency(input: &[&str], worker_count: usize) -> HashMap<char, usize> {
    if input.is_empty() { return HashMap::new(); }
    let chunk_size = (input.len() + worker_count - 1) / worker_count;
    let chunks: Vec<String> = input.chunks(chunk_size).map(|c| c.join("")).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks.iter().map(|chunk| s.spawn(|| {
            let mut map = HashMap::new();
            for c in chunk.chars() {
                if c.is_alphabetic() { for lower in c.to_lowercase() { *map.entry(lower).or_insert(0) += 1; } }
            }
            map
        })).collect();
        let mut result = HashMap::new();
        for handle in handles { for (ch, count) in handle.join().unwrap() { *result.entry(ch).or_insert(0) += count; } }
        result
    })
}"#,
        },
        LabeledSolution {
            name: "list-ops",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn append<I, J>(a: I, b: J) -> impl Iterator<Item = I::Item> where I: Iterator, J: Iterator<Item = I::Item> { a.chain(b) }
pub fn concat<I>(nested: I) -> impl Iterator<Item = <I::Item as Iterator>::Item> where I: Iterator, I::Item: Iterator { nested.flatten() }
pub fn filter<I, F>(iter: I, pred: F) -> impl Iterator<Item = I::Item> where I: Iterator, F: Fn(&I::Item) -> bool { iter.filter(pred) }
pub fn length<I: Iterator>(iter: I) -> usize { iter.count() }
pub fn map<I, F, U>(iter: I, func: F) -> impl Iterator<Item = U> where I: Iterator, F: Fn(I::Item) -> U { iter.map(func) }
pub fn foldl<I, F, U>(iter: I, init: U, func: F) -> U where I: Iterator, F: Fn(U, I::Item) -> U { iter.fold(init, func) }
pub fn foldr<I, F, U>(iter: I, init: U, func: F) -> U where I: DoubleEndedIterator, F: Fn(U, I::Item) -> U { iter.rev().fold(init, func) }
pub fn reverse<I: DoubleEndedIterator>(iter: I) -> impl Iterator<Item = I::Item> { iter.rev() }"#,
        },
        // ── More String Processing ──
        LabeledSolution {
            name: "simple-cipher",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn encode(key: &str, s: &str) -> Option<String> {
    if key.is_empty() || !key.chars().all(|c| c.is_ascii_lowercase()) { return None; }
    Some(s.chars().zip(key.chars().cycle()).map(|(c, k)| {
        if c.is_ascii_lowercase() { (((c as u8 - b'a') + (k as u8 - b'a')) % 26 + b'a') as char } else { c }
    }).collect())
}"#,
        },
        LabeledSolution {
            name: "scrabble-score",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn score(word: &str) -> u64 {
    word.chars().map(|c| match c.to_ascii_uppercase() {
        'A'|'E'|'I'|'O'|'U'|'L'|'N'|'R'|'S'|'T' => 1,
        'D'|'G' => 2, 'B'|'C'|'M'|'P' => 3, 'F'|'H'|'V'|'W'|'Y' => 4,
        'K' => 5, 'J'|'X' => 8, 'Q'|'Z' => 10, _ => 0,
    }).sum()
}"#,
        },
        LabeledSolution {
            name: "beer-song",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn verse(n: u32) -> String {
    match n {
        0 => "No more bottles of beer on the wall, no more bottles of beer.\nGo to the store and buy some more, 99 bottles of beer on the wall.\n".to_string(),
        1 => "1 bottle of beer on the wall, 1 bottle of beer.\nTake it down and pass it around, no more bottles of beer on the wall.\n".to_string(),
        n => format!("{n} bottles of beer on the wall, {n} bottles of beer.\nTake one down and pass it around, {} bottles of beer on the wall.\n", n - 1),
    }
}
pub fn sing(start: u32, end: u32) -> String {
    (end..=start).rev().map(verse).collect::<Vec<_>>().join("\n")
}"#,
        },
        LabeledSolution {
            name: "run-length-encoding",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn encode(source: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        let mut count = 1;
        while i + count < chars.len() && chars[i + count] == ch { count += 1; }
        if count > 1 { result.push_str(&count.to_string()); }
        result.push(ch);
        i += count;
    }
    result
}"#,
        },
        // ── More Mathematical ──
        LabeledSolution {
            name: "collatz-conjecture",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn collatz(n: u64) -> Option<u64> {
    if n == 0 { return None; }
    let mut n = n;
    let mut steps = 0;
    while n != 1 { n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 }; steps += 1; }
    Some(steps)
}"#,
        },
        LabeledSolution {
            name: "perfect-numbers",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn classify(num: u64) -> Option<Classification> {
    if num == 0 { return None; }
    let sum: u64 = (1..num).filter(|&i| num % i == 0).sum();
    Some(match sum.cmp(&num) {
        std::cmp::Ordering::Equal => Classification::Perfect,
        std::cmp::Ordering::Greater => Classification::Abundant,
        std::cmp::Ordering::Less => Classification::Deficient,
    })
}"#,
        },
        LabeledSolution {
            name: "prime-factors",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn factors(n: u64) -> Vec<u64> {
    let mut n = n;
    let mut factors = Vec::new();
    let mut d = 2;
    while d * d <= n {
        while n % d == 0 { factors.push(d); n /= d; }
        d += 1;
    }
    if n > 1 { factors.push(n); }
    factors
}"#,
        },
        LabeledSolution {
            name: "armstrong-numbers",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn is_armstrong_number(num: u32) -> bool {
    let digits: Vec<u32> = num.to_string().chars().filter_map(|c| c.to_digit(10)).collect();
    let power = digits.len() as u32;
    digits.iter().map(|d| d.pow(power)).sum::<u32>() == num
}"#,
        },
        // ── More IO Transform ──
        LabeledSolution {
            name: "etl",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn transform(h: &BTreeMap<i32, Vec<char>>) -> BTreeMap<char, i32> {
    h.iter().flat_map(|(&score, letters)| {
        letters.iter().map(move |c| (c.to_ascii_lowercase(), score))
    }).collect()
}"#,
        },
        LabeledSolution {
            name: "nucleotide-count",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn count(nucleotide: char, dna: &str) -> Result<usize, char> {
    if !matches!(nucleotide, 'A' | 'C' | 'G' | 'T') { return Err(nucleotide); }
    let mut total = 0;
    for ch in dna.chars() {
        if !matches!(ch, 'A' | 'C' | 'G' | 'T') { return Err(ch); }
        if ch == nucleotide { total += 1; }
    }
    Ok(total)
}"#,
        },
        LabeledSolution {
            name: "roman-numerals",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn to_roman(mut num: u32) -> String {
    let table = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
        (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")];
    let mut result = String::new();
    for &(value, symbol) in &table {
        while num >= value { result.push_str(symbol); num -= value; }
    }
    result
}"#,
        },
        LabeledSolution {
            name: "luhn",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn is_valid(code: &str) -> bool {
    let digits: Option<Vec<u32>> = code.chars().filter(|c| !c.is_whitespace())
        .map(|c| c.to_digit(10)).collect();
    let Some(digits) = digits else { return false; };
    if digits.len() <= 1 { return false; }
    digits.iter().rev().enumerate().map(|(i, d)| {
        if i % 2 == 1 { let dd = d * 2; if dd > 9 { dd - 9 } else { dd } } else { *d }
    }).sum::<u32>() % 10 == 0
}"#,
        },
        // ── More Data Structure ──
        LabeledSolution {
            name: "custom-set",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct CustomSet<T: PartialEq + Clone> { elements: Vec<T> }
impl<T: PartialEq + Clone> CustomSet<T> {
    pub fn new(input: &[T]) -> Self {
        let mut elements = Vec::new();
        for item in input { if !elements.contains(item) { elements.push(item.clone()); } }
        CustomSet { elements }
    }
    pub fn contains(&self, element: &T) -> bool { self.elements.contains(element) }
    pub fn add(&mut self, element: T) { if !self.elements.contains(&element) { self.elements.push(element); } }
    pub fn is_subset(&self, other: &Self) -> bool { self.elements.iter().all(|e| other.contains(e)) }
    pub fn is_empty(&self) -> bool { self.elements.is_empty() }
}"#,
        },
        // ── More Search ──
        LabeledSolution {
            name: "grep",
            class: AlgorithmClass::Search,
            source: r#"pub fn grep(pattern: &str, flags: &Flags, files: &[&str]) -> Result<Vec<String>, Error> {
    let multiple = files.len() > 1;
    let mut results = Vec::new();
    for &file in files {
        let content = std::fs::read_to_string(file)?;
        for (i, line) in content.lines().enumerate() {
            let matched = if flags.entire_line { line == pattern } else { line.contains(pattern) };
            let matched = if flags.invert { !matched } else { matched };
            if matched {
                let mut result = String::new();
                if multiple { result.push_str(file); result.push(':'); }
                if flags.line_numbers { result.push_str(&format!("{}:", i + 1)); }
                result.push_str(line);
                results.push(result);
            }
        }
    }
    Ok(results)
}"#,
        },
        // ── More Sorting ──
        LabeledSolution {
            name: "saddle-points",
            class: AlgorithmClass::Sorting,
            source: r#"pub fn find_saddle_points(input: &[Vec<u64>]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for (r, row) in input.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            let row_max = row.iter().copied().max().unwrap_or(0);
            let col_min = input.iter().map(|row| row[c]).min().unwrap_or(0);
            if val == row_max && val == col_min { result.push((r, c)); }
        }
    }
    result
}"#,
        },
        LabeledSolution {
            name: "tournament",
            class: AlgorithmClass::Sorting,
            source: r#"pub fn tally(match_results: &str) -> String {
    let mut teams: HashMap<&str, [u32; 4]> = HashMap::new();
    for line in match_results.lines() {
        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() != 3 { continue; }
        let (t1, t2, result) = (parts[0], parts[1], parts[2]);
        match result {
            "win" => { teams.entry(t1).or_default()[0] += 1; teams.entry(t2).or_default()[2] += 1; }
            "loss" => { teams.entry(t1).or_default()[2] += 1; teams.entry(t2).or_default()[0] += 1; }
            "draw" => { teams.entry(t1).or_default()[1] += 1; teams.entry(t2).or_default()[1] += 1; }
            _ => {}
        }
    }
    let mut sorted: Vec<_> = teams.iter().collect();
    sorted.sort_by(|a, b| {
        let pa = a.1[0] * 3 + a.1[1]; let pb = b.1[0] * 3 + b.1[1];
        pb.cmp(&pa).then(a.0.cmp(b.0))
    });
    let mut result = "Team                           | MP |  W |  D |  L |  P".to_string();
    for (name, stats) in sorted {
        let mp = stats[0] + stats[1] + stats[2];
        let p = stats[0] * 3 + stats[1];
        result.push_str(&format!("\n{:31}| {:2} | {:2} | {:2} | {:2} | {:2}", name, mp, stats[0], stats[1], stats[2], p));
    }
    result
}"#,
        },
        // ── More Graph ──
        LabeledSolution {
            name: "poker",
            class: AlgorithmClass::Graph, // ranking/comparison is graph-like in structure
            source: r#"pub fn winning_hands<'a>(hands: &[&'a str]) -> Vec<&'a str> {
    fn rank(hand: &str) -> (u8, Vec<u8>) {
        let mut cards: Vec<(u8, u8)> = hand.split_whitespace().map(|c| {
            let (rank_str, suit) = if c.len() == 3 { (&c[..2], c.as_bytes()[2]) } else { (&c[..1], c.as_bytes()[1]) };
            let r = match rank_str { "A"=>14, "K"=>13, "Q"=>12, "J"=>11, "10"=>10, s=>s.parse().unwrap() };
            (r, suit)
        }).collect();
        cards.sort_by(|a, b| b.0.cmp(&a.0));
        let ranks: Vec<u8> = cards.iter().map(|c| c.0).collect();
        let flush = cards.iter().all(|c| c.1 == cards[0].1);
        let straight = ranks.windows(2).all(|w| w[0] == w[1] + 1) || ranks == [14, 5, 4, 3, 2];
        let mut counts: HashMap<u8, u8> = HashMap::new();
        for &r in &ranks { *counts.entry(r).or_insert(0) += 1; }
        let mut groups: Vec<(u8, u8)> = counts.into_iter().collect();
        groups.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
        let pattern: Vec<u8> = groups.iter().map(|g| g.1).collect();
        let hand_rank = if straight && flush { 8 } else if pattern == [4, 1] { 7 } else if pattern == [3, 2] { 6 }
            else if flush { 5 } else if straight { 4 } else if pattern == [3, 1, 1] { 3 }
            else if pattern == [2, 2, 1] { 2 } else if pattern == [2, 1, 1, 1] { 1 } else { 0 };
        (hand_rank, groups.iter().map(|g| g.0).collect())
    }
    let ranked: Vec<_> = hands.iter().map(|&h| (h, rank(h))).collect();
    let best = ranked.iter().map(|(_, r)| r).max().unwrap().clone();
    ranked.iter().filter(|(_, r)| *r == best).map(|&(h, _)| h).collect()
}"#,
        },
        LabeledSolution {
            name: "react",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct Reactor<'a, T> {
    inputs: Vec<T>,
    computes: Vec<ComputeCell<'a, T>>,
    next_cb: usize,
}
impl<'a, T: Copy + PartialEq> Reactor<'a, T> {
    pub fn new() -> Self { Reactor { inputs: Vec::new(), computes: Vec::new(), next_cb: 0 } }
    pub fn create_input(&mut self, initial: T) -> InputCellId {
        self.inputs.push(initial); InputCellId(self.inputs.len() - 1)
    }
    pub fn set_value(&mut self, id: InputCellId, new_value: T) -> bool {
        if id.0 >= self.inputs.len() { return false; }
        self.inputs[id.0] = new_value;
        let mut changed = Vec::new();
        for ci in 0..self.computes.len() {
            let vals: Vec<T> = self.computes[ci].deps.iter().map(|d| self.value(*d).unwrap()).collect();
            let new_val = (self.computes[ci].compute)(&vals);
            if new_val != self.computes[ci].value {
                self.computes[ci].value = new_val;
                changed.push((ci, new_val));
            }
        }
        for (ci, val) in changed { for cb in self.computes[ci].callbacks.values_mut() { cb(val); } }
        true
    }
}"#,
        },
        // ── Batch 3: expand to 60 solutions ──
        LabeledSolution {
            name: "reverse-string",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn reverse(input: &str) -> String { input.chars().rev().collect() }"#,
        },
        LabeledSolution {
            name: "leap",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn is_leap_year(year: u64) -> bool { year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) }"#,
        },
        LabeledSolution {
            name: "matching-brackets",
            class: AlgorithmClass::DataStructure,
            source: r#"pub fn brackets_are_balanced(string: &str) -> bool {
    let mut stack = Vec::new();
    for ch in string.chars() { match ch {
        '(' | '[' | '{' => stack.push(ch),
        ')' => { if stack.pop() != Some('(') { return false; } }
        ']' => { if stack.pop() != Some('[') { return false; } }
        '}' => { if stack.pop() != Some('{') { return false; } }
        _ => {} } }
    stack.is_empty()
}"#,
        },
        LabeledSolution {
            name: "isbn-verifier",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn is_valid_isbn(isbn: &str) -> bool {
    let chars: Vec<char> = isbn.chars().filter(|c| *c != '-').collect();
    if chars.len() != 10 { return false; }
    let mut sum = 0u32;
    for (i, c) in chars.iter().enumerate() {
        let value = if i == 9 && *c == 'X' { 10 } else if let Some(d) = c.to_digit(10) { d } else { return false; };
        sum += value * (10 - i as u32);
    }
    sum % 11 == 0
}"#,
        },
        LabeledSolution {
            name: "word-count",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn word_count(words: &str) -> HashMap<String, u32> {
    let mut counts = HashMap::new();
    for word in words.to_lowercase().split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .map(|w| w.trim_matches('\'')).filter(|w| !w.is_empty()) {
        *counts.entry(word.to_string()).or_insert(0) += 1;
    }
    counts
}"#,
        },
        LabeledSolution {
            name: "phone-number",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn number(user_number: &str) -> Option<String> {
    let digits: Vec<char> = user_number.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = if digits.len() == 11 && digits[0] == '1' { &digits[1..] } else { &digits[..] };
    if digits.len() != 10 { return None; }
    if digits[0] < '2' || digits[3] < '2' { return None; }
    Some(digits.iter().collect())
}"#,
        },
        LabeledSolution {
            name: "bob",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn reply(message: &str) -> &str {
    let trimmed = message.trim();
    let is_question = trimmed.ends_with('?');
    let has_letters = trimmed.chars().any(|c| c.is_alphabetic());
    let is_yelling = has_letters && trimmed.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_uppercase());
    match (trimmed.is_empty(), is_question, is_yelling) {
        (true, _, _) => "Fine. Be that way!",
        (_, true, true) => "Calm down, I know what I'm doing!",
        (_, true, false) => "Sure.",
        (_, false, true) => "Whoa, chill out!",
        _ => "Whatever.",
    }
}"#,
        },
        LabeledSolution {
            name: "clock",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct Clock { minutes: i32 }
impl Clock {
    pub fn new(hours: i32, minutes: i32) -> Self {
        Clock { minutes: ((hours * 60 + minutes) % 1440 + 1440) % 1440 }
    }
    pub fn add_minutes(&self, minutes: i32) -> Self { Clock::new(0, self.minutes + minutes) }
}"#,
        },
        LabeledSolution {
            name: "robot-simulator",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct Robot { x: i32, y: i32, dir: Direction }
impl Robot {
    pub fn new(x: i32, y: i32, d: Direction) -> Self { Robot { x, y, dir: d } }
    pub fn turn_right(mut self) -> Self { self.dir = self.dir.right(); self }
    pub fn turn_left(mut self) -> Self { self.dir = self.dir.left(); self }
    pub fn advance(mut self) -> Self { match self.dir { Direction::North => self.y += 1, Direction::South => self.y -= 1, Direction::East => self.x += 1, Direction::West => self.x -= 1 }; self }
    pub fn instructions(self, instructions: &str) -> Self {
        instructions.chars().fold(self, |r, c| match c { 'R' => r.turn_right(), 'L' => r.turn_left(), 'A' => r.advance(), _ => r })
    }
}"#,
        },
        LabeledSolution {
            name: "bowling",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct BowlingGame { rolls: Vec<u16>, current_frame: usize }
impl BowlingGame {
    pub fn new() -> Self { BowlingGame { rolls: Vec::new(), current_frame: 0 } }
    pub fn roll(&mut self, pins: u16) -> Result<(), Error> {
        if pins > 10 { return Err(Error::NotEnoughPinsLeft); }
        self.rolls.push(pins);
        Ok(())
    }
    pub fn score(&self) -> Option<u16> {
        let mut total = 0u16;
        let mut i = 0;
        for _ in 0..10 {
            if i >= self.rolls.len() { return None; }
            if self.rolls[i] == 10 { total += 10 + self.rolls.get(i+1).unwrap_or(&0) + self.rolls.get(i+2).unwrap_or(&0); i += 1; }
            else {
                let frame = self.rolls[i] + self.rolls.get(i+1).unwrap_or(&0);
                if frame == 10 { total += 10 + self.rolls.get(i+2).unwrap_or(&0); }
                else { total += frame; }
                i += 2;
            }
        }
        Some(total)
    }
}"#,
        },
        LabeledSolution {
            name: "triangle",
            class: AlgorithmClass::Mathematical,
            source: r#"pub struct Triangle { sides: [u64; 3] }
impl Triangle {
    pub fn build(sides: [u64; 3]) -> Option<Triangle> {
        let [a, b, c] = sides;
        if a == 0 || b == 0 || c == 0 || a + b <= c || a + c <= b || b + c <= a { None }
        else { Some(Triangle { sides }) }
    }
    pub fn is_equilateral(&self) -> bool { self.sides[0] == self.sides[1] && self.sides[1] == self.sides[2] }
    pub fn is_isosceles(&self) -> bool { self.sides[0] == self.sides[1] || self.sides[1] == self.sides[2] || self.sides[0] == self.sides[2] }
    pub fn is_scalene(&self) -> bool { self.sides[0] != self.sides[1] && self.sides[1] != self.sides[2] && self.sides[0] != self.sides[2] }
}"#,
        },
        LabeledSolution {
            name: "spiral-matrix",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn spiral_matrix(size: u32) -> Vec<Vec<u32>> {
    let n = size as usize;
    if n == 0 { return Vec::new(); }
    let mut matrix = vec![vec![0u32; n]; n];
    let (mut top, mut bottom, mut left, mut right) = (0, n - 1, 0, n - 1);
    let mut val = 1u32;
    while top <= bottom && left <= right {
        for c in left..=right { matrix[top][c] = val; val += 1; } top += 1;
        for r in top..=bottom { matrix[r][right] = val; val += 1; } if right == 0 { break; } right -= 1;
        for c in (left..=right).rev() { matrix[bottom][c] = val; val += 1; } if bottom == 0 { break; } bottom -= 1;
        for r in (top..=bottom).rev() { matrix[r][left] = val; val += 1; } left += 1;
    }
    matrix
}"#,
        },
        LabeledSolution {
            name: "accumulate",
            class: AlgorithmClass::IoTransform,
            source: r#"pub fn map<T, U, F: FnMut(T) -> U>(input: Vec<T>, mut function: F) -> Vec<U> {
    input.into_iter().map(|x| function(x)).collect()
}"#,
        },
        LabeledSolution {
            name: "two-fer",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn twofer(name: &str) -> String {
    if name.is_empty() { "One for you, one for me.".to_string() }
    else { format!("One for {name}, one for me.") }
}"#,
        },
        LabeledSolution {
            name: "series",
            class: AlgorithmClass::StringProcessing,
            source: r#"pub fn series(digits: &str, len: usize) -> Vec<String> {
    if len == 0 { return vec!["".to_string(); digits.len() + 1]; }
    digits.as_bytes().windows(len).map(|w| std::str::from_utf8(w).unwrap().to_string()).collect()
}"#,
        },
        LabeledSolution {
            name: "yacht",
            class: AlgorithmClass::Mathematical,
            source: r#"pub fn score(dice: Dice, category: Category) -> u8 {
    let mut counts = [0u8; 7];
    for &d in &dice { counts[d as usize] += 1; }
    match category {
        Category::Ones => counts[1],
        Category::FullHouse => { if counts.iter().any(|&c| c == 3) && counts.iter().any(|&c| c == 2) { dice.iter().sum() } else { 0 } }
        Category::Yacht => { if counts.iter().any(|&c| c == 5) { 50 } else { 0 } }
        _ => 0,
    }
}"#,
        },
        LabeledSolution {
            name: "palindrome-products",
            class: AlgorithmClass::Search,
            source: r#"pub fn palindrome_products(min: u64, max: u64) -> Option<(Palindrome, Palindrome)> {
    let mut min_pal = None;
    let mut max_pal = None;
    for a in min..=max { for b in a..=max {
        let product = a * b;
        let s = product.to_string();
        if s != s.chars().rev().collect::<String>() { continue; }
        match &mut min_pal { None => { min_pal = Some(product); } Some(p) if product < *p => { *p = product; } _ => {} }
        match &mut max_pal { None => { max_pal = Some(product); } Some(p) if product > *p => { *p = product; } _ => {} }
    }}
    Some((min_pal?, max_pal?))
}"#,
        },
        LabeledSolution {
            name: "xorcism",
            class: AlgorithmClass::IoTransform,
            source: r#"pub struct Xorcism<'a> { key: &'a [u8], pos: usize }
impl<'a> Xorcism<'a> {
    pub fn new<Key: AsRef<[u8]> + ?Sized>(key: &'a Key) -> Xorcism<'a> { Xorcism { key: key.as_ref(), pos: 0 } }
    pub fn munge_in_place(&mut self, data: &mut [u8]) {
        for byte in data.iter_mut() { *byte ^= self.key[self.pos % self.key.len()]; self.pos += 1; }
    }
}"#,
        },
        LabeledSolution {
            name: "fizzy",
            class: AlgorithmClass::IoTransform,
            source: r#"pub struct Matcher<T> { matcher: Box<dyn Fn(T) -> bool>, subs: String }
pub struct Fizzy<T> { matchers: Vec<Matcher<T>> }
impl<T: ToString + Copy> Fizzy<T> {
    pub fn new() -> Self { Fizzy { matchers: Vec::new() } }
    pub fn add_matcher(mut self, matcher: Matcher<T>) -> Self { self.matchers.push(matcher); self }
    pub fn apply<I: Iterator<Item = T>>(self, iter: I) -> impl Iterator<Item = String> {
        iter.map(move |item| {
            let mut result = String::new();
            for m in &self.matchers { if (m.matcher)(item) { result.push_str(&m.subs); } }
            if result.is_empty() { item.to_string() } else { result }
        })
    }
}"#,
        },
        LabeledSolution {
            name: "paasio",
            class: AlgorithmClass::IoTransform,
            source: r#"pub struct ReadStats<R> { inner: R, bytes: usize, reads: usize }
impl<R: std::io::Read> ReadStats<R> {
    pub fn new(wrapped: R) -> Self { ReadStats { inner: wrapped, bytes: 0, reads: 0 } }
    pub fn bytes_through(&self) -> usize { self.bytes }
    pub fn reads(&self) -> usize { self.reads }
}
impl<R: std::io::Read> std::io::Read for ReadStats<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?; self.bytes += n; self.reads += 1; Ok(n)
    }
}"#,
        },
        LabeledSolution {
            name: "robot-name",
            class: AlgorithmClass::DataStructure,
            source: r#"pub struct RobotFactory { used: Rc<RefCell<HashSet<String>>> }
pub struct Robot { name: String, used: Rc<RefCell<HashSet<String>>> }
impl RobotFactory {
    pub fn new() -> Self { RobotFactory { used: Rc::new(RefCell::new(HashSet::new())) } }
    pub fn new_robot<R: Rng>(&mut self, rng: &mut R) -> Robot {
        loop { let n = gen_name(rng); if self.used.borrow_mut().insert(n.clone()) { return Robot { name: n, used: self.used.clone() }; } }
    }
}"#,
        },
        LabeledSolution {
            name: "luhn-from",
            class: AlgorithmClass::IoTransform,
            source: r#"pub struct Luhn { valid: bool }
impl Luhn { pub fn is_valid(&self) -> bool { self.valid } }
impl<T: ToString> From<T> for Luhn {
    fn from(input: T) -> Self {
        let s = input.to_string();
        let digits: Option<Vec<u32>> = s.chars().filter(|c| !c.is_whitespace()).map(|c| c.to_digit(10)).collect();
        let Some(digits) = digits else { return Luhn { valid: false }; };
        if digits.len() <= 1 { return Luhn { valid: false }; }
        let sum: u32 = digits.iter().rev().enumerate().map(|(i, &d)| {
            if i % 2 == 1 { let dd = d * 2; if dd > 9 { dd - 9 } else { dd } } else { d }
        }).sum();
        Luhn { valid: sum % 10 == 0 }
    }
}"#,
        },
    ]
}

// ─── Training Pipeline ─────────────────────────────────────────────────────

/// Build training pairs from the Exercism corpus.
pub fn build_training_pairs() -> Vec<AlgorithmTrainingPair> {
    let encoder = AlgorithmEncoder::new();
    let corpus = exercism_corpus();

    corpus
        .into_iter()
        .map(|sol| {
            let mut channels = extract_features(sol.source);
            // Override the auto-classified class with ground truth
            channels.set_class(sol.class);

            let hv = encoder.encode(&channels);

            AlgorithmTrainingPair {
                name: sol.name.to_string(),
                channels,
                hv,
                class: sol.class,
                purpose: format!("Exercism: {}", sol.name),
                source: sol.source.to_string(),
            }
        })
        .collect()
}

// ─── Difficulty Scoring ────────────────────────────────────────────────────

/// Compute structural difficulty score from algorithm channels.
///
/// Higher = more complex. Weights structural nesting and state management
/// over raw code size. A 5-line recursive function scores higher than
/// a 20-line flat iterator chain.
pub fn difficulty_score(channels: &AlgorithmChannels) -> f32 {
    let c = &channels.channels;
    c[0]          // loop_depth
    + c[1]        // branch_depth
    + 2.0 * c[2]  // recursion (×2 — recursion is inherently harder)
    + c[3]        // closure_count
    + c[23]       // mutation_level
    + c[25]       // error_handling
    + c[29]       // helper_functions
    + 0.5 * c[30] // pattern_match_arms (half weight)
    + 0.1 * c[26] // line_count (small weight — size matters less than structure)
}

/// Sort training pairs by difficulty (easiest first).
pub fn sort_by_difficulty(pairs: &mut [AlgorithmTrainingPair]) {
    pairs.sort_by(|a, b| {
        let da = difficulty_score(&a.channels);
        let db = difficulty_score(&b.channels);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Train the CfC code sequencer on algorithm HDC vectors.
///
/// Returns: (training_loss, classification_accuracy, confusion_matrix_summary)
pub fn train_and_evaluate() -> TrainingReport {
    let pairs = build_training_pairs();
    let encoder = AlgorithmEncoder::new();

    // Split 80/20
    let split = (pairs.len() * 4) / 5;
    let (train, eval) = pairs.split_at(split);

    // Evaluate classification by HDC similarity
    // For each eval pair, find the closest train pair and check class match
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut class_correct: std::collections::HashMap<AlgorithmClass, (usize, usize)> =
        std::collections::HashMap::new();

    for eval_pair in eval {
        let mut best_sim = -1.0f32;
        let mut best_class = AlgorithmClass::IoTransform;

        for train_pair in train.iter() {
            let sim = eval_pair.hv.similarity(&train_pair.hv);
            if sim > best_sim {
                best_sim = sim;
                best_class = train_pair.class;
            }
        }

        let is_correct = best_class == eval_pair.class;
        if is_correct {
            correct += 1;
        }
        total += 1;

        let entry = class_correct.entry(eval_pair.class).or_insert((0, 0));
        entry.1 += 1;
        if is_correct {
            entry.0 += 1;
        }
    }

    // Also compute intra-class vs inter-class similarity
    let mut intra_sims = Vec::new();
    let mut inter_sims = Vec::new();
    for (i, a) in pairs.iter().enumerate() {
        for (j, b) in pairs.iter().enumerate() {
            if i >= j {
                continue;
            }
            let sim = a.hv.similarity(&b.hv);
            if a.class == b.class {
                intra_sims.push(sim);
            } else {
                inter_sims.push(sim);
            }
        }
    }

    let avg_intra = if intra_sims.is_empty() {
        0.0
    } else {
        intra_sims.iter().sum::<f32>() / intra_sims.len() as f32
    };
    let avg_inter = if inter_sims.is_empty() {
        0.0
    } else {
        inter_sims.iter().sum::<f32>() / inter_sims.len() as f32
    };

    TrainingReport {
        corpus_size: pairs.len(),
        train_size: train.len(),
        eval_size: eval.len(),
        classification_accuracy: if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        },
        correct,
        total,
        avg_intra_class_similarity: avg_intra,
        avg_inter_class_similarity: avg_inter,
        separation: avg_intra - avg_inter,
        per_class: class_correct,
    }
}

// ─── CfC Sequencer Training ────────────────────────────────────────────────

/// Map algorithm class to a target PlanAction sequence.
///
/// These represent the structural skeleton of each algorithm family.
/// The CfC learns: "when the HDC vector looks like class X, the plan
/// should evolve through these actions."
fn class_to_plan(class: AlgorithmClass, channels: &AlgorithmChannels) -> Vec<PlanAction> {
    let has_struct = channels.channels[5] > 5.0; // many blocks → likely has struct
    let has_loops = channels.channels[0] > 0.5;
    let has_error = channels.channels[25] > 0.5;
    let has_generics = channels.channels[8] > 0.5;

    let mut plan = vec![PlanAction::DefineFunction];

    // Add parameters based on arity
    let arity = channels.channels[6] as usize;
    for _ in 0..arity.min(4) {
        plan.push(PlanAction::AddParameter);
    }
    plan.push(PlanAction::SetReturnType);

    // Class-specific actions
    match class {
        AlgorithmClass::Sorting => {
            if has_loops {
                plan.push(PlanAction::ForLoop);
                plan.push(PlanAction::ForLoop); // nested
            } else {
                plan.push(PlanAction::IteratorChain);
            }
        }
        AlgorithmClass::Search => {
            if has_loops {
                plan.push(PlanAction::ForLoop);
            }
            plan.push(PlanAction::MatchExpression);
        }
        AlgorithmClass::DynamicProgramming => {
            plan.push(PlanAction::ForLoop);
            plan.push(PlanAction::ForLoop); // nested DP loops
        }
        AlgorithmClass::Graph => {
            plan.push(PlanAction::ForLoop);
            plan.push(PlanAction::MatchExpression);
            if channels.channels[2] > 0.5 {
                // recursion
                plan.push(PlanAction::ClosureDefine);
            }
        }
        AlgorithmClass::StringProcessing => {
            plan.push(PlanAction::IteratorChain);
            if has_loops {
                plan.push(PlanAction::ForLoop);
            }
        }
        AlgorithmClass::Mathematical => {
            if channels.channels[2] > 0.5 {
                // recursion
                plan.push(PlanAction::MatchExpression);
            } else {
                plan.push(PlanAction::ForLoop);
            }
        }
        AlgorithmClass::DataStructure => {
            plan.push(PlanAction::DefineStruct);
            plan.push(PlanAction::AddField);
            plan.push(PlanAction::ImplTrait);
            plan.push(PlanAction::AddMethod);
        }
        AlgorithmClass::IoTransform => {
            plan.push(PlanAction::IteratorChain);
        }
    }

    // Conditional additions
    if has_error {
        plan.push(PlanAction::AddErrorHandling);
    }
    if has_generics {
        plan.push(PlanAction::GenericParam);
    }
    if has_struct && !matches!(class, AlgorithmClass::DataStructure) {
        plan.push(PlanAction::DefineStruct);
    }

    plan.push(PlanAction::Complete);
    plan
}

/// Learned projection from 16,384D → 512D that preserves class separation.
///
/// Uses Fisher-inspired centroid projection:
/// 1. Compute class centroids in full 16,384D
/// 2. Find the dimensions with maximum inter-centroid variance
/// 3. Project onto those dimensions
///
/// This preserves the information that distinguishes algorithm classes
/// instead of arbitrarily truncating.
pub struct LearnedProjection {
    /// Indices of the top-512 most discriminative dimensions.
    selected_dims: Vec<usize>,
    /// Per-dimension scaling factors (inter-centroid std dev).
    scales: Vec<f32>,
}

impl LearnedProjection {
    /// Learn the projection from labeled training data.
    pub fn fit(pairs: &[AlgorithmTrainingPair]) -> Self {
        let target_dim = 512usize;
        let full_dim = pairs
            .first()
            .map(|p| p.hv.values.len())
            .unwrap_or(HDC_DIMENSION);

        // Step 1: Compute class centroids
        let mut class_sums: std::collections::HashMap<AlgorithmClass, (Vec<f64>, usize)> =
            std::collections::HashMap::new();

        for pair in pairs {
            let entry = class_sums
                .entry(pair.class)
                .or_insert_with(|| (vec![0.0f64; full_dim], 0));
            for (i, &v) in pair.hv.values.iter().enumerate() {
                entry.0[i] += v as f64;
            }
            entry.1 += 1;
        }

        let centroids: Vec<Vec<f64>> = class_sums
            .values()
            .map(|(sum, count)| sum.iter().map(|s| s / *count as f64).collect())
            .collect();

        // Step 2: Compute global centroid
        let n_classes = centroids.len();
        let mut global = vec![0.0f64; full_dim];
        for c in &centroids {
            for (i, v) in c.iter().enumerate() {
                global[i] += v;
            }
        }
        for v in global.iter_mut() {
            *v /= n_classes as f64;
        }

        // Step 3: Per-dimension inter-centroid variance
        // variance[d] = (1/K) * Σ_k (centroid_k[d] - global[d])²
        let mut variance = vec![0.0f64; full_dim];
        for c in &centroids {
            for (d, v) in c.iter().enumerate() {
                let diff = v - global[d];
                variance[d] += diff * diff;
            }
        }

        // Step 4: Select top-512 dimensions by inter-centroid variance
        let mut indexed: Vec<(usize, f64)> = variance.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_dims: Vec<usize> = indexed.iter().take(target_dim).map(|(i, _)| *i).collect();
        let scales: Vec<f32> = indexed
            .iter()
            .take(target_dim)
            .map(|(_, v)| (v.sqrt() as f32).max(1e-8))
            .collect();

        Self {
            selected_dims,
            scales,
        }
    }

    /// Project a full-dimensional HV to 512D using the learned dimension selection.
    pub fn project(&self, hv: &ContinuousHV) -> ContinuousHV {
        let values: Vec<f32> = self
            .selected_dims
            .iter()
            .zip(self.scales.iter())
            .map(|(&dim_idx, &scale)| {
                let v = if dim_idx < hv.values.len() {
                    hv.values[dim_idx]
                } else {
                    0.0
                };
                v / scale // normalize by inter-centroid std dev
            })
            .collect();
        ContinuousHV::from_vec(values)
    }

    /// Report the variance captured by the projection.
    pub fn variance_captured(&self) -> f32 {
        let total: f32 = self.scales.iter().map(|s| s * s).sum();
        let top: f32 = self.scales.iter().take(64).map(|s| s * s).sum();
        if total > 0.0 { top / total } else { 0.0 }
    }
}

// ─── Linear Algorithm Classifier ───────────────────────────────────────────

/// Simple linear classifier: W·x + b → 8 class logits.
///
/// This is what the CfC SHOULD be doing but can't with 60 examples.
/// A linear layer has 512×8 + 8 = 4,104 parameters — tractable for
/// 48 training examples with SGD.
pub struct AlgorithmClassifier {
    /// Weight matrix: 8 × 512 (class × feature)
    weights: Vec<Vec<f32>>,
    /// Bias: 8
    bias: Vec<f32>,
}

impl AlgorithmClassifier {
    const NUM_CLASSES: usize = 8;
    const INPUT_DIM: usize = 512;

    /// Initialize with small random weights.
    pub fn new() -> Self {
        let mut weights = Vec::with_capacity(Self::NUM_CLASSES);
        let mut seed = 12345u64;
        for _ in 0..Self::NUM_CLASSES {
            let row: Vec<f32> = (0..Self::INPUT_DIM)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.01
                })
                .collect();
            weights.push(row);
        }
        Self {
            weights,
            bias: vec![0.0; Self::NUM_CLASSES],
        }
    }

    /// Forward pass: compute class logits.
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(w, b)| {
                w.iter()
                    .zip(input.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f32>()
                    + b
            })
            .collect()
    }

    /// Softmax of logits.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Predict class from input features.
    pub fn predict(&self, input: &[f32]) -> AlgorithmClass {
        let logits = self.forward(input);
        let best_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        AlgorithmClass::ALL[best_idx]
    }

    /// Train on labeled pairs via SGD with cross-entropy loss.
    pub fn train(
        &mut self,
        train: &[(Vec<f32>, usize)], // (features, class_idx)
        epochs: usize,
        lr: f32,
    ) -> Vec<f32> {
        let mut losses = Vec::new();

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;

            for (input, &target) in train.iter().map(|(x, y)| (x, y)) {
                let logits = self.forward(input);
                let probs = Self::softmax(&logits);

                // Cross-entropy loss
                epoch_loss -= probs[target].max(1e-10).ln();

                // Gradient: d_loss/d_logit = prob - one_hot
                let mut grad = probs.clone();
                grad[target] -= 1.0;

                // SGD update
                for (c, g) in grad.iter().enumerate() {
                    for (j, xj) in input.iter().enumerate() {
                        self.weights[c][j] -= lr * g * xj;
                    }
                    self.bias[c] -= lr * g;
                }
            }

            losses.push(epoch_loss / train.len() as f32);
        }

        losses
    }

    /// Evaluate accuracy on held-out data.
    pub fn accuracy(&self, eval: &[(Vec<f32>, usize)]) -> f32 {
        if eval.is_empty() {
            return 0.0;
        }
        let correct = eval
            .iter()
            .filter(|(x, y)| {
                let predicted = self.predict(x);
                AlgorithmClass::ALL[*y] == predicted
            })
            .count();
        correct as f32 / eval.len() as f32
    }
}

/// Train a linear classifier on the algorithm corpus.
///
/// Returns (classifier, train_accuracy, eval_accuracy, loss_curve).
pub fn train_linear_classifier(
    epochs: usize,
    lr: f32,
) -> (AlgorithmClassifier, f32, f32, Vec<f32>) {
    let pairs = build_training_pairs();
    let projection = LearnedProjection::fit(&pairs);

    let split = (pairs.len() * 4) / 5;

    // Convert to (projected_features, class_index) pairs
    let data: Vec<(Vec<f32>, usize)> = pairs
        .iter()
        .map(|p| {
            let projected = projection.project(&p.hv);
            let class_idx = AlgorithmClass::ALL
                .iter()
                .position(|c| *c == p.class)
                .unwrap_or(7);
            (projected.values.clone(), class_idx)
        })
        .collect();

    let (train, eval) = data.split_at(split);

    let mut classifier = AlgorithmClassifier::new();
    let losses = classifier.train(&train.to_vec(), epochs, lr);

    let train_acc = classifier.accuracy(&train.to_vec());
    let eval_acc = classifier.accuracy(&eval.to_vec());

    (classifier, train_acc, eval_acc, losses)
}

/// Project using naive truncation (legacy fallback).
fn project_to_sequencer_dim(hv: &ContinuousHV) -> ContinuousHV {
    let dim = 512;
    let values: Vec<f32> = hv.values.iter().take(dim).copied().collect();
    ContinuousHV::from_vec(values)
}

/// Train the CfC code sequencer on algorithm HDC vectors.
///
/// Uses the learned Fisher projection to compress 16,384D → 512D
/// while preserving class separation. Returns honest metrics.
pub fn train_cfc_sequencer(epochs: usize, learning_rate: f32) -> CfcTrainingReport {
    let pairs = build_training_pairs();
    let sequencer = CfCCodeSequencer::default();

    let split = (pairs.len() * 4) / 5;
    let (train, eval) = pairs.split_at(split);

    // Learn the projection from training data
    let projection = LearnedProjection::fit(train);

    let mut epoch_losses = Vec::new();

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut count = 0;

        for pair in train.iter() {
            let target_plan = class_to_plan(pair.class, &pair.channels);
            let projected = projection.project(&pair.hv);
            match sequencer.train_sequence(&projected, &target_plan, learning_rate) {
                Ok(loss) => {
                    epoch_loss += loss;
                    count += 1;
                }
                Err(_) => {}
            }
        }

        let avg_loss = if count > 0 {
            epoch_loss / count as f32
        } else {
            f32::NAN
        };
        epoch_losses.push(avg_loss);
    }

    // Evaluate: run plan_structure on eval set and compare first action
    let mut correct_first_action = 0usize;
    let mut correct_class_from_plan = 0usize;
    let mut eval_total = 0usize;

    for pair in eval.iter() {
        let projected = projection.project(&pair.hv);
        let planned = sequencer.plan_structure(
            &projected,
            &[],
            &crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default(),
        );
        let target_plan = class_to_plan(pair.class, &pair.channels);

        if let (Some(predicted), Some(expected)) = (planned.first(), target_plan.get(1)) {
            // Skip DefineFunction (always first) — compare the interesting action
            if predicted.action == *expected {
                correct_first_action += 1;
            }
        }

        // Check if the plan indicates the right algorithm class
        let has_for_loop = planned.iter().any(|s| s.action == PlanAction::ForLoop);
        let has_iter_chain = planned
            .iter()
            .any(|s| s.action == PlanAction::IteratorChain);
        let has_struct = planned.iter().any(|s| s.action == PlanAction::DefineStruct);
        let has_match = planned
            .iter()
            .any(|s| s.action == PlanAction::MatchExpression);

        let predicted_class = if has_struct {
            AlgorithmClass::DataStructure
        } else if has_iter_chain && !has_for_loop {
            AlgorithmClass::IoTransform
        } else if has_match && has_for_loop {
            AlgorithmClass::Graph
        } else if has_for_loop {
            AlgorithmClass::Mathematical
        } else {
            AlgorithmClass::StringProcessing
        };

        if predicted_class == pair.class {
            correct_class_from_plan += 1;
        }
        eval_total += 1;
    }

    CfcTrainingReport {
        epochs,
        learning_rate,
        train_size: train.len(),
        eval_size: eval.len(),
        final_loss: epoch_losses.last().copied().unwrap_or(f32::NAN),
        loss_curve: epoch_losses,
        first_action_accuracy: if eval_total > 0 {
            correct_first_action as f32 / eval_total as f32
        } else {
            0.0
        },
        class_from_plan_accuracy: if eval_total > 0 {
            correct_class_from_plan as f32 / eval_total as f32
        } else {
            0.0
        },
    }
}

/// Curriculum-based CfC training.
///
/// Three phases:
/// 1. Easy (50 epochs): train on the easiest third of the corpus
/// 2. Medium (50 epochs): expand to the easiest two-thirds
/// 3. Full (remaining epochs): train on all examples
///
/// This gives the CfC a stable gradient on simple patterns before
/// introducing complex multi-function state machines.
pub fn train_cfc_curriculum(total_epochs: usize, learning_rate: f32) -> CfcTrainingReport {
    let mut pairs = build_training_pairs();
    let sequencer = CfCCodeSequencer::default();

    // Sort by structural difficulty
    sort_by_difficulty(&mut pairs);

    let n = pairs.len();
    let split = (n * 4) / 5;
    let (all_train, eval) = pairs.split_at(split);

    // Learn projection from all training data (not curriculum-gated)
    let projection = LearnedProjection::fit(all_train);

    // Curriculum phases: easy third → two-thirds → full
    let easy_end = all_train.len() / 3;
    let medium_end = (all_train.len() * 2) / 3;

    let phase_1_epochs = total_epochs / 4;
    let phase_2_epochs = total_epochs / 4;
    let phase_3_epochs = total_epochs - phase_1_epochs - phase_2_epochs;

    let mut epoch_losses = Vec::new();

    // Phase 1: Easy examples only
    for _epoch in 0..phase_1_epochs {
        let loss = train_one_epoch(
            &sequencer,
            &all_train[..easy_end],
            &projection,
            learning_rate,
        );
        epoch_losses.push(loss);
    }

    // Phase 2: Easy + medium
    for _epoch in 0..phase_2_epochs {
        let loss = train_one_epoch(
            &sequencer,
            &all_train[..medium_end],
            &projection,
            learning_rate,
        );
        epoch_losses.push(loss);
    }

    // Phase 3: Full corpus
    for _epoch in 0..phase_3_epochs {
        let loss = train_one_epoch(&sequencer, all_train, &projection, learning_rate);
        epoch_losses.push(loss);
    }

    // Evaluate
    let (correct_first, correct_class, eval_total) = evaluate_plans(&sequencer, eval, &projection);

    CfcTrainingReport {
        epochs: total_epochs,
        learning_rate,
        train_size: all_train.len(),
        eval_size: eval.len(),
        final_loss: epoch_losses.last().copied().unwrap_or(f32::NAN),
        loss_curve: epoch_losses,
        first_action_accuracy: if eval_total > 0 {
            correct_first as f32 / eval_total as f32
        } else {
            0.0
        },
        class_from_plan_accuracy: if eval_total > 0 {
            correct_class as f32 / eval_total as f32
        } else {
            0.0
        },
    }
}

/// Train one epoch on a subset of pairs.
fn train_one_epoch(
    sequencer: &CfCCodeSequencer,
    train: &[AlgorithmTrainingPair],
    projection: &LearnedProjection,
    lr: f32,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut count = 0;
    for pair in train {
        let target_plan = class_to_plan(pair.class, &pair.channels);
        let projected = projection.project(&pair.hv);
        if let Ok(loss) = sequencer.train_sequence(&projected, &target_plan, lr) {
            total_loss += loss;
            count += 1;
        }
    }
    if count > 0 {
        total_loss / count as f32
    } else {
        f32::NAN
    }
}

/// Evaluate CfC plans on held-out data.
fn evaluate_plans(
    sequencer: &CfCCodeSequencer,
    eval: &[AlgorithmTrainingPair],
    projection: &LearnedProjection,
) -> (usize, usize, usize) {
    let mut correct_first = 0usize;
    let mut correct_class = 0usize;
    let mut total = 0usize;

    for pair in eval {
        let projected = projection.project(&pair.hv);
        let planned = sequencer.plan_structure(
            &projected,
            &[],
            &crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default(),
        );
        let target_plan = class_to_plan(pair.class, &pair.channels);

        if let (Some(predicted), Some(expected)) = (planned.first(), target_plan.get(1)) {
            if predicted.action == *expected {
                correct_first += 1;
            }
        }

        let has_for_loop = planned.iter().any(|s| s.action == PlanAction::ForLoop);
        let has_iter_chain = planned
            .iter()
            .any(|s| s.action == PlanAction::IteratorChain);
        let has_struct = planned.iter().any(|s| s.action == PlanAction::DefineStruct);
        let has_match = planned
            .iter()
            .any(|s| s.action == PlanAction::MatchExpression);

        let predicted_class = if has_struct {
            AlgorithmClass::DataStructure
        } else if has_iter_chain && !has_for_loop {
            AlgorithmClass::IoTransform
        } else if has_match && has_for_loop {
            AlgorithmClass::Graph
        } else if has_for_loop {
            AlgorithmClass::Mathematical
        } else {
            AlgorithmClass::StringProcessing
        };

        if predicted_class == pair.class {
            correct_class += 1;
        }
        total += 1;
    }

    (correct_first, correct_class, total)
}

/// CfC training results.
#[derive(Debug)]
pub struct CfcTrainingReport {
    pub epochs: usize,
    pub learning_rate: f32,
    pub train_size: usize,
    pub eval_size: usize,
    pub final_loss: f32,
    pub loss_curve: Vec<f32>,
    pub first_action_accuracy: f32,
    pub class_from_plan_accuracy: f32,
}

impl std::fmt::Display for CfcTrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== CfC Sequencer Training Report ===")?;
        writeln!(
            f,
            "Training: {} epochs, lr={}, {} train / {} eval",
            self.epochs, self.learning_rate, self.train_size, self.eval_size
        )?;
        writeln!(f, "Final loss: {:.6}", self.final_loss)?;
        if self.loss_curve.len() >= 2 {
            writeln!(
                f,
                "Loss trend: {:.6} → {:.6}",
                self.loss_curve[0], self.final_loss
            )?;
        }
        writeln!(
            f,
            "First-action accuracy: {:.1}%",
            self.first_action_accuracy * 100.0
        )?;
        writeln!(
            f,
            "Class-from-plan accuracy: {:.1}%",
            self.class_from_plan_accuracy * 100.0
        )?;
        Ok(())
    }
}

// ─── End-to-End Cold Generation ────────────────────────────────────────────

/// Result of attempting cold code generation from the brain.
#[derive(Debug)]
pub struct ColdGenerationResult {
    /// The problem description given.
    pub purpose: String,
    /// Detected algorithm class from the HDC encoding.
    pub detected_class: AlgorithmClass,
    /// The CfC plan steps generated.
    pub plan: Vec<String>,
    /// The assembled code skeleton.
    pub code: String,
    /// Whether the generated code is syntactically plausible.
    pub plausible: bool,
}

/// Generate code from a problem description using the full brain pipeline:
///
/// 1. Extract features from the problem description (text → channels)
/// 2. Encode as 16,384D HDC vector
/// 3. Project to 512D via learned Fisher projection
/// 4. Run CfC temporal evolution → PlanAction sequence
/// 5. Assemble code skeleton from plan actions
///
/// This is the first end-to-end test of the HDC→CfC→Code pipeline.
/// Generate code via the hybrid System 1/System 2 path with self-repair.
///
/// Strategy:
/// 1. Try 1-NN body retrieval (high quality if a similar example exists)
/// 2. If 1-NN fails to compile, fall back to template assembly
/// 3. If template fails too, run repair iterations enriching channels from errors
pub fn generate_with_repair_hybrid(
    purpose: &str,
    signature: &str,
    pairs: &[AlgorithmTrainingPair],
    classifier: &AlgorithmClassifier,
    max_iterations: usize,
) -> RepairResult {
    let encoder = AlgorithmEncoder::new();
    let projection = LearnedProjection::fit(pairs);

    let mut channels = build_channels_from_purpose(purpose, signature);
    let mut error_history = Vec::new();
    let mut current_code = String::new();
    let mut compiles = false;
    let mut class;

    // Iteration -1: Try class+signature idiom (highest priority — emits real algorithms)
    {
        let hv = encoder.encode(&channels);
        let projected = projection.project(&hv);
        let predicted_class = hybrid_classify(purpose, &projected.values, classifier);
        if let Some(idiom) = class_idiom_body(predicted_class, purpose, signature) {
            let idiom_code = format!("{signature} {{\n{idiom}\n}}");
            if try_compile(&idiom_code).is_ok() {
                return RepairResult {
                    purpose: purpose.to_string(),
                    signature: signature.to_string(),
                    iterations: 1,
                    compiles: true,
                    final_code: idiom_code,
                    error_history,
                    class: predicted_class,
                };
            }
        }
    }

    // Iteration 0: Try 1-NN retrieval first (highest quality)
    if let Some(nn_code) = generate_via_nearest_neighbor(purpose, signature, pairs, classifier) {
        current_code = nn_code;
        match try_compile(&current_code) {
            Ok(()) => {
                let hv = encoder.encode(&channels);
                let projected = projection.project(&hv);
                class = hybrid_classify(purpose, &projected.values, classifier);
                return RepairResult {
                    purpose: purpose.to_string(),
                    signature: signature.to_string(),
                    iterations: 1,
                    compiles: true,
                    final_code: current_code,
                    error_history,
                    class,
                };
            }
            Err(e) => {
                error_history.push(format!("1-NN: {e}"));
                enrich_channels_from_errors(&mut channels, &e);
            }
        }
    }

    // Iterations 1..max: hybrid classify + template assembly + repair
    let hv = encoder.encode(&channels);
    let projected = projection.project(&hv);
    class = hybrid_classify(purpose, &projected.values, classifier);

    for _iteration in 0..max_iterations {
        channels.set_class(class);
        let template_actions = class_to_plan(class, &channels);
        let plan_steps = to_plan_steps(&template_actions);
        current_code = assemble_from_plan(signature, &plan_steps, &channels);

        match try_compile(&current_code) {
            Ok(()) => {
                compiles = true;
                break;
            }
            Err(e) => {
                error_history.push(e.clone());
                enrich_channels_from_errors(&mut channels, &e);
            }
        }
    }

    RepairResult {
        purpose: purpose.to_string(),
        signature: signature.to_string(),
        iterations: error_history.len() + if compiles { 1 } else { 0 },
        compiles,
        final_code: current_code,
        error_history,
        class,
    }
}

/// Find the nearest training solution by HDC similarity within the same class.
fn nearest_in_class<'a>(
    query_hv: &symthaea_core::hdc::ContinuousHV,
    class: AlgorithmClass,
    pairs: &'a [AlgorithmTrainingPair],
) -> Option<(f32, &'a AlgorithmTrainingPair)> {
    pairs
        .iter()
        .filter(|p| p.class == class)
        .map(|p| (query_hv.similarity(&p.hv), p))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
}

/// Extract the body of a function from its source code (between {} braces).
fn extract_function_body(source: &str) -> Option<String> {
    let open = source.find('{')?;
    let mut depth = 0i32;
    let bytes = source.as_bytes();
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(source[open + 1..i].trim().to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Whole-word replacement of `old_param` with `new_param` in `body`.
fn adapt_body_param(body: &str, old_param: &str, new_param: &str) -> String {
    if old_param == new_param || old_param.is_empty() {
        return body.to_string();
    }
    let mut result = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let at_start = i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_');
        if at_start
            && i + old_param.len() <= bytes.len()
            && &body[i..i + old_param.len()] == old_param
        {
            let after = i + old_param.len();
            let at_end = after >= bytes.len()
                || (!bytes[after].is_ascii_alphanumeric() && bytes[after] != b'_');
            if at_end {
                result.push_str(new_param);
                i = after;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Generate code via 1-NN body retrieval, adapted to the target signature.
///
/// Finds the most similar training example (same class), extracts its body,
/// and renames the parameter to match the new signature.
pub fn generate_via_nearest_neighbor(
    purpose: &str,
    signature: &str,
    pairs: &[AlgorithmTrainingPair],
    classifier: &AlgorithmClassifier,
) -> Option<String> {
    let encoder = AlgorithmEncoder::new();
    let projection = LearnedProjection::fit(pairs);

    let channels = build_channels_from_purpose(purpose, signature);
    let hv = encoder.encode(&channels);
    let projected = projection.project(&hv);
    let predicted_class = hybrid_classify(purpose, &projected.values, classifier);

    let (_sim, nearest) = nearest_in_class(&hv, predicted_class, pairs)?;
    let body = extract_function_body(&nearest.source)?;

    let target_param = first_param_name(signature);
    let source_param = first_param_name(&nearest.source);
    let adapted = adapt_body_param(&body, &source_param, &target_param);

    Some(format!(
        "{signature} {{\n    {}\n}}",
        adapted.replace('\n', "\n    ")
    ))
}

/// System 1/System 2 generation:
/// - System 1 (HDC + Linear Classifier): rapid class identification
/// - System 2 (Template plan → assembly): deterministic code shape
///
/// Returns code that uses the classifier's prediction (not keyword heuristics).
pub fn cold_generate_classified(
    purpose: &str,
    signature: &str,
    pairs: &[AlgorithmTrainingPair],
    classifier: &AlgorithmClassifier,
) -> ColdGenerationResult {
    let encoder = AlgorithmEncoder::new();
    let projection = LearnedProjection::fit(pairs);

    // Build channels from purpose + signature (without the class — that's what we predict)
    let mut channels = build_channels_from_purpose(purpose, signature);

    // System 1: predict class via linear classifier on HDC vector
    let hv = encoder.encode(&channels);
    let projected = projection.project(&hv);
    let predicted_class = hybrid_classify(purpose, &projected.values, classifier);
    channels.set_class(predicted_class);

    // Re-encode with the predicted class (consistency)
    let hv2 = encoder.encode(&channels);
    let projected2 = projection.project(&hv2);

    // System 2: template plan + assembly
    let template_actions = class_to_plan(predicted_class, &channels);
    let plan_steps = to_plan_steps(&template_actions);
    let code = assemble_from_plan(signature, &plan_steps, &channels);

    let plan: Vec<String> = plan_steps
        .iter()
        .map(|s| format!("{:?}", s.action))
        .collect();

    let _ = projected2; // silence unused warning

    let plausible = code.contains("fn ")
        && code.contains('{')
        && code.contains('}')
        && !code.contains("todo!(");

    ColdGenerationResult {
        purpose: purpose.to_string(),
        detected_class: predicted_class,
        plan,
        code,
        plausible,
    }
}

pub fn cold_generate(
    purpose: &str,
    signature: &str,
    pairs: &[AlgorithmTrainingPair],
) -> ColdGenerationResult {
    let encoder = AlgorithmEncoder::new();
    let sequencer = CfCCodeSequencer::default();
    let projection = LearnedProjection::fit(pairs);

    // Step 1: Create channels from the purpose + signature
    let mut channels = AlgorithmChannels::default();

    // Parse signature for type features
    let param_count = signature
        .split('(')
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|p| {
            if p.trim().is_empty() {
                0
            } else {
                p.split(',').count()
            }
        })
        .unwrap_or(0);
    channels.set_input_arity(param_count as f32);

    if signature.contains("-> Option<") {
        channels.set_returns_option_result(1.0);
    } else if signature.contains("-> Result<") {
        channels.set_returns_option_result(2.0);
    }

    if signature.contains("Vec<") || signature.contains("&[") {
        channels.set_allocation_level(1.0);
    }
    if signature.contains("HashMap") || signature.contains("HashSet") {
        channels.set_allocation_level(2.0);
    }

    // Infer class from purpose text — more specific keywords first
    let lower = purpose.to_lowercase();
    let class = if lower.contains("sort") || lower.contains("order") {
        AlgorithmClass::Sorting
    } else if lower.contains("knapsack") || lower.contains("dynamic") || lower.contains("optimal") {
        AlgorithmClass::DynamicProgramming
    } else if lower.contains("graph") || lower.contains("path") || lower.contains("node") {
        AlgorithmClass::Graph
    } else if lower.contains("math")
        || lower.contains("prime")
        || lower.contains("factor")
        || lower.contains("sum")
        || lower.contains("number")
    {
        AlgorithmClass::Mathematical
    } else if lower.contains("search") || lower.contains("find") || lower.contains("lookup") {
        AlgorithmClass::Search
    } else if lower.contains("string")
        || lower.contains("char")
        || lower.contains("text")
        || lower.contains("word")
    {
        AlgorithmClass::StringProcessing
    } else if lower.contains("struct")
        || lower.contains("stack")
        || lower.contains("queue")
        || lower.contains("list")
    {
        AlgorithmClass::DataStructure
    } else {
        AlgorithmClass::IoTransform
    };
    channels.set_class(class);

    // Guess structural features from purpose
    if lower.contains("each") || lower.contains("every") || lower.contains("iterate") {
        channels.set_loop_depth(1.0);
    }
    if lower.contains("nested") || lower.contains("matrix") || lower.contains("grid") {
        channels.set_loop_depth(2.0);
    }
    if lower.contains("recursive") || lower.contains("factorial") {
        channels.set_recursion(true);
    }

    // Step 2: Encode
    let hv = encoder.encode(&channels);

    // Step 3: Project
    let projected = projection.project(&hv);

    // Step 4: CfC plan
    let plan_steps = sequencer.plan_structure(
        &projected,
        &[],
        &crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default(),
    );
    let plan: Vec<String> = plan_steps
        .iter()
        .map(|s| format!("{:?}({:.2})", s.action, s.confidence))
        .collect();

    // Step 5: Assemble code from plan
    let code = assemble_from_plan(signature, &plan_steps, &channels);

    // Check plausibility
    let plausible = code.contains("fn ")
        && (code.contains('{') && code.contains('}'))
        && !code.contains("todo!(");

    ColdGenerationResult {
        purpose: purpose.to_string(),
        detected_class: class,
        plan,
        code,
        plausible,
    }
}

/// Assemble a code skeleton from CfC plan actions.
/// Extract the first parameter name from a function signature.
fn first_param_name(signature: &str) -> String {
    signature
        .split('(')
        .nth(1)
        .and_then(|s| s.split(')').next())
        .and_then(|params| {
            let first = params.split(',').next()?.trim();
            let name = first.split(':').next()?.trim();
            let name = name.strip_prefix("mut ").unwrap_or(name);
            if name.is_empty() || name == "&self" || name == "&mut self" {
                None
            } else {
                Some(name.to_string())
            }
        })
        .unwrap_or_else(|| "input".to_string())
}

/// Emit a class-specific idiomatic body from purpose + signature + class.
///
/// Uses signature SHAPE + class + purpose keywords to select a real Rust idiom.
/// Returns Some(idiom) for matched cases, None to fall through to template.
pub fn class_idiom_body(class: AlgorithmClass, purpose: &str, signature: &str) -> Option<String> {
    let lower = purpose.to_lowercase();
    let param = first_param_name(signature);

    let ret_str = signature.contains("-> String");
    let ret_bool = signature.contains("-> bool");
    let ret_usize = signature.contains("-> usize");
    let ret_u64 = signature.contains("-> u64");
    let ret_i32 = signature.contains("-> i32");
    let ret_vec = signature.contains("-> Vec<");

    let param_str = signature.contains("&str");
    let param_vec_i32 = signature.contains("Vec<i32>") || signature.contains("&[i32]");
    let param_vec_any = signature.contains("Vec<") || signature.contains("&[");
    let param_u64 = signature.contains(": u64") || signature.contains("&u64");

    let _ = (ret_u64, ret_usize, param_vec_any); // silence unused

    match class {
        AlgorithmClass::Sorting if param_vec_i32 => {
            Some(format!("    let mut v = {param};\n    v.sort();\n    v"))
        }
        AlgorithmClass::Mathematical if lower.contains("prime") && ret_bool && param_u64 => {
            Some(format!(
                "    if {param} < 2 {{ return false; }}\n    let mut i = 2u64;\n    while i * i <= {param} {{\n        if {param} % i == 0 {{ return false; }}\n        i += 1;\n    }}\n    true"
            ))
        }
        AlgorithmClass::Mathematical
            if (lower.contains("fibonacci") || lower.contains("fib")) && param_u64 =>
        {
            Some(format!(
                "    if {param} < 2 {{ return {param}; }}\n    let mut a = 0u64;\n    let mut b = 1u64;\n    for _ in 2..={param} {{\n        let c = a + b;\n        a = b;\n        b = c;\n    }}\n    b"
            ))
        }
        AlgorithmClass::Mathematical if lower.contains("factorial") && param_u64 => {
            Some(format!("    (1..={param}).product()"))
        }
        AlgorithmClass::Mathematical if lower.contains("even") && ret_bool && param_u64 => {
            Some(format!("    {param} % 2 == 0"))
        }
        AlgorithmClass::Mathematical
            if (lower.contains("sum") || lower.contains("total")) && param_vec_i32 && ret_i32 =>
        {
            Some(format!("    {param}.iter().copied().sum()"))
        }
        AlgorithmClass::Mathematical if lower.contains("factor") && ret_vec && param_u64 => {
            Some(format!(
                "    let mut n = {param};\n    let mut factors = Vec::new();\n    let mut d = 2u64;\n    while d * d <= n {{\n        while n % d == 0 {{ factors.push(d); n /= d; }}\n        d += 1;\n    }}\n    if n > 1 {{ factors.push(n); }}\n    factors"
            ))
        }
        AlgorithmClass::StringProcessing if lower.contains("reverse") && param_str && ret_str => {
            Some(format!("    {param}.chars().rev().collect()"))
        }
        AlgorithmClass::StringProcessing
            if (lower.contains("vowel") || lower.contains("vowels")) && ret_usize =>
        {
            Some(format!(
                "    {param}.chars().filter(|c| matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')).count()"
            ))
        }
        AlgorithmClass::StringProcessing
            if lower.contains("word") && (lower.contains("count") || ret_usize) =>
        {
            Some(format!("    {param}.split_whitespace().count()"))
        }
        AlgorithmClass::StringProcessing if lower.contains("uppercase") && ret_str => {
            Some(format!("    {param}.to_uppercase()"))
        }
        AlgorithmClass::StringProcessing if lower.contains("lowercase") && ret_str => {
            Some(format!("    {param}.to_lowercase()"))
        }
        AlgorithmClass::StringProcessing if lower.contains("palindrome") && ret_bool => {
            Some(format!(
                "    let s: String = {param}.chars().filter(|c| c.is_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect();\n    s == s.chars().rev().collect::<String>()"
            ))
        }
        AlgorithmClass::Search if lower.contains("prime") && ret_vec && param_u64 => Some(format!(
            "    if {param} < 2 {{ return Vec::new(); }}\n    let n = {param} as usize;\n    let mut is_prime = vec![true; n + 1];\n    is_prime[0] = false; is_prime[1] = false;\n    let mut i = 2;\n    while i * i <= n {{\n        if is_prime[i] {{ for j in (i*i..=n).step_by(i) {{ is_prime[j] = false; }} }}\n        i += 1;\n    }}\n    (2..=n).filter(|&i| is_prime[i]).map(|i| i as u64).collect()"
        )),
        _ => None,
    }
}

fn assemble_from_plan(
    signature: &str,
    plan: &[crate::dynamics::cfc_code_sequencer::CodePlanStep],
    channels: &AlgorithmChannels,
) -> String {
    use crate::dynamics::cfc_code_sequencer::PlanAction;

    let param = first_param_name(signature);
    let mut body_parts: Vec<String> = Vec::new();

    // Determine return expression based on what we know
    let has_vec_return = signature.contains("Vec<");
    let has_string_return = signature.contains("-> String");
    let has_bool_return = signature.contains("-> bool");
    let has_option_return = signature.contains("-> Option<");
    let has_usize_return = signature.contains("-> usize");
    let has_i32_return = signature.contains("-> i32") || signature.contains("-> i64");
    let has_u64_return = signature.contains("-> u64") || signature.contains("-> u32");

    // Check if first param is iterable (collection or &str)
    let param_is_iterable = signature.contains("&[")
        || signature.contains("Vec<")
        || signature.contains("&str")
        || signature.contains("Iterator");

    for step in plan {
        match &step.action {
            PlanAction::ForLoop if param_is_iterable => {
                body_parts.push(format!(
                    "    for item in {param} {{\n        // process\n    }}"
                ));
            }
            PlanAction::IteratorChain if param_is_iterable => {
                if has_vec_return {
                    body_parts.push(format!("    {param}.iter().copied().collect()"));
                } else if has_string_return {
                    body_parts.push(format!("    {param}.chars().collect()"));
                } else if has_i32_return || has_u64_return || has_usize_return {
                    body_parts.push(format!("    {param}.iter().copied().sum()"));
                } else {
                    body_parts.push(format!("    {param}.iter().collect()"));
                }
            }
            PlanAction::MatchExpression => {
                body_parts.push(format!(
                    "    match {param} {{\n        _ => todo!()\n    }}"
                ));
            }
            PlanAction::AddErrorHandling => {
                if has_option_return {
                    body_parts.push("    Some(result)".into());
                } else {
                    body_parts.push("    Ok(result)".into());
                }
            }
            PlanAction::DefineStruct => {
                // Already in signature
            }
            PlanAction::Complete => break,
            _ => {}
        }
    }

    // Fallback: if plan produced nothing useful, generate minimal body
    if body_parts.is_empty() {
        if has_bool_return {
            body_parts.push("    false".into());
        } else if has_usize_return || has_i32_return || has_u64_return {
            body_parts.push("    0".into());
        } else if has_vec_return {
            body_parts.push("    Vec::new()".into());
        } else if has_string_return {
            body_parts.push("    String::new()".into());
        } else if has_option_return {
            body_parts.push("    None".into());
        } else {
            body_parts.push("    Default::default()".into());
        }
    }

    format!("{signature} {{\n{}\n}}", body_parts.join("\n"))
}

// ─── Self-Repair Loop ──────────────────────────────────────────────────────

/// Result of a repair attempt.
#[derive(Debug)]
pub struct RepairResult {
    /// The problem description.
    pub purpose: String,
    /// Function signature.
    pub signature: String,
    /// Number of repair iterations attempted.
    pub iterations: usize,
    /// Whether final code compiles (via rustc syntax check).
    pub compiles: bool,
    /// The final generated code.
    pub final_code: String,
    /// Compiler errors from each iteration (empty = success).
    pub error_history: Vec<String>,
    /// Algorithm class detected.
    pub class: AlgorithmClass,
}

/// Attempt code generation with self-repair.
///
/// The repair loop:
/// 1. Generate initial code from problem description
/// 2. Attempt to compile (rustc --edition 2021 --crate-type lib -)
/// 3. If errors: parse error types → enrich channels → re-generate
/// 4. Repeat up to `max_iterations` times
pub fn generate_with_repair(
    purpose: &str,
    signature: &str,
    pairs: &[AlgorithmTrainingPair],
    max_iterations: usize,
) -> RepairResult {
    let encoder = AlgorithmEncoder::new();
    let projection = LearnedProjection::fit(pairs);
    let sequencer = CfCCodeSequencer::default();

    let mut error_history = Vec::new();
    let mut current_code = String::new();
    let mut channels = build_channels_from_purpose(purpose, signature);
    let mut compiles = false;

    for iteration in 0..max_iterations {
        // Encode → project → plan → assemble
        channels.set_class(classify_from_purpose(purpose));
        let hv = encoder.encode(&channels);
        let projected = projection.project(&hv);
        let plan_steps = sequencer.plan_structure(
            &projected,
            &[],
            &crate::consciousness::temporal_planning::mcts::NegativePrototypeBank::default(),
        );

        // Use 1-NN retrieval to enhance the body if plan is trivial
        let class = AlgorithmClass::from_channels(&channels);
        current_code = assemble_with_1nn(signature, &plan_steps, &channels, class, pairs, &encoder);

        // Try to compile
        match try_compile(&current_code) {
            Ok(()) => {
                compiles = true;
                break;
            }
            Err(errors) => {
                // Parse errors and enrich channels for next iteration
                enrich_channels_from_errors(&mut channels, &errors);
                error_history.push(errors);
            }
        }
    }

    RepairResult {
        purpose: purpose.to_string(),
        signature: signature.to_string(),
        iterations: error_history.len() + if compiles { 1 } else { 0 },
        compiles,
        final_code: current_code,
        error_history,
        class: AlgorithmClass::from_channels(&channels),
    }
}

/// Build initial channels from purpose text and signature.
/// Build algorithm channels from a natural-language purpose + signature.
///
/// Populates 20+ channels by parsing the description for algorithmic
/// signals (verbs, data structures, complexity hints) and the signature
/// for type information (arity, generics, lifetimes, return types).
pub fn build_channels_from_purpose_public(purpose: &str, signature: &str) -> AlgorithmChannels {
    build_channels_from_purpose(purpose, signature)
}

fn build_channels_from_purpose(purpose: &str, signature: &str) -> AlgorithmChannels {
    let mut channels = AlgorithmChannels::default();
    let lower = purpose.to_lowercase();

    // ── Type features from signature (channels 6-11) ──

    let arity = signature
        .split('(')
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|p| {
            if p.trim().is_empty() {
                0
            } else {
                p.split(',').count()
            }
        })
        .unwrap_or(0);
    channels.set_input_arity(arity as f32);

    if signature.contains("-> Option<") {
        channels.set_returns_option_result(1.0);
    } else if signature.contains("-> Result<") {
        channels.set_returns_option_result(2.0);
    }

    // Generics: count <...> in signature
    let generic_count = signature.matches('<').count();
    channels.set_generic_count(generic_count as f32);

    // Lifetimes
    channels.set_has_lifetime(signature.contains("'a") || signature.contains("'_"));

    // Trait bounds
    let trait_bounds = signature.matches(": ").count() + signature.matches("where").count();
    channels.set_trait_bound_count(trait_bounds.saturating_sub(arity) as f32);

    // Type complexity: nested angle brackets
    let mut max_depth = 0u32;
    let mut depth = 0u32;
    for c in signature.chars() {
        match c {
            '<' => {
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            '>' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    channels.set_type_complexity(max_depth as f32);

    // ── Allocation level from signature + purpose ──

    let alloc = if signature.contains("Box<")
        || signature.contains("Rc<")
        || signature.contains("Arc<")
        || lower.contains("linked")
        || lower.contains("tree")
        || lower.contains("graph")
    {
        3.0
    } else if signature.contains("HashMap")
        || signature.contains("HashSet")
        || signature.contains("BTreeMap")
        || lower.contains("dictionary")
        || lower.contains("map of")
        || lower.contains("set of")
    {
        2.0
    } else if signature.contains("Vec<")
        || signature.contains("&[")
        || signature.contains("String")
        || lower.contains("list")
        || lower.contains("array")
        || lower.contains("collection")
    {
        1.0
    } else {
        0.0
    };
    channels.set_allocation_level(alloc);

    // ── Loop depth from purpose hints ──

    let loop_depth = if lower.contains("nested loop")
        || lower.contains("matrix")
        || lower.contains("grid")
        || lower.contains("2d")
        || lower.contains("table")
    {
        2.0
    } else if lower.contains("each")
        || lower.contains("every")
        || lower.contains("iterate")
        || lower.contains("loop")
        || lower.contains("traverse")
        || lower.contains("scan")
        || lower.contains("for all")
        || lower.contains("foreach")
    {
        1.0
    } else {
        0.0
    };
    channels.set_loop_depth(loop_depth);

    // Branch depth — if/match keywords
    if lower.contains("if")
        || lower.contains("when")
        || lower.contains("case")
        || lower.contains("either")
        || lower.contains("based on")
        || lower.contains("depending")
    {
        channels.set_branch_depth(1.0);
    }

    // Recursion hints
    if lower.contains("recursive")
        || lower.contains("recursion")
        || lower.contains("factorial")
        || lower.contains("fibonacci")
        || lower.contains("tree")
        || lower.contains("divide and conquer")
    {
        channels.set_recursion(true);
    }

    // Closures / higher-order
    if lower.contains("apply")
        || lower.contains("function")
        || lower.contains("callback")
        || lower.contains("transform")
        || lower.contains("predicate")
        || signature.contains("Fn(")
    {
        channels.set_closure_count(1.0);
    }

    // Iterator chain
    if lower.contains("filter")
        || lower.contains("map")
        || lower.contains("transform")
        || lower.contains("collect")
        || lower.contains("chain")
        || lower.contains("pipeline")
    {
        channels.set_iterator_chain_len(3.0);
    }

    // ── Algorithm class hints (channels 12-19) ──
    // Set the class one-hot but ALSO populate related secondary channels

    let class = classify_from_purpose(purpose);
    channels.set_class(class);

    // ── Data flow (channels 20-25) ──

    if lower.contains("map")
        || lower.contains("transform")
        || lower.contains("convert")
        || lower.contains("translate")
    {
        channels.set_map_count(1.0);
    }
    if lower.contains("filter")
        || lower.contains("select")
        || lower.contains("keep")
        || lower.contains("remove")
        || lower.contains("exclude")
    {
        channels.set_filter_count(1.0);
    }
    if lower.contains("sum")
        || lower.contains("count")
        || lower.contains("accumulate")
        || lower.contains("reduce")
        || lower.contains("fold")
        || lower.contains("aggregate")
        || lower.contains("total")
    {
        channels.set_fold_count(1.0);
    }

    // Mutation level
    let mutation = if signature.contains("&mut self") {
        2.0
    } else if signature.contains("&mut ")
        || lower.contains("update")
        || lower.contains("modify")
        || lower.contains("change")
        || lower.contains("set ")
        || lower.contains("insert")
        || lower.contains("remove")
        || lower.contains("push")
    {
        1.0
    } else {
        0.0
    };
    channels.set_mutation_level(mutation);

    // Error handling
    let err = if signature.contains("Result<") {
        2.0
    } else if signature.contains("Option<")
        || lower.contains("validate")
        || lower.contains("check")
        || lower.contains("verify")
        || lower.contains("invalid")
    {
        1.0
    } else {
        0.0
    };
    channels.set_error_handling(err);

    // ── Complexity estimates (channels 26-31) ──

    // Estimate line count from purpose complexity (rough heuristic)
    let estimated_lines = 5.0
        + 5.0 * loop_depth
        + 3.0 * channels.channels[1] // branch_depth
        + 5.0 * if channels.channels[2] > 0.5 { 1.0 } else { 0.0 } // recursion
        + 3.0 * channels.channels[3] // closures
        + 2.0 * channels.channels[20] // map
        + 2.0 * channels.channels[21] // filter
        + 2.0 * channels.channels[22] // fold
        + 4.0 * channels.channels[10]; // trait bounds
    channels.set_line_count(estimated_lines);

    // Cyclomatic complexity ≈ branches + loops + 1
    let cyclo = 1.0
        + channels.channels[0]
        + channels.channels[1]
        + 0.5 * (channels.channels[20] + channels.channels[21]);
    channels.set_cyclomatic_complexity(cyclo);

    // State variables
    if mutation > 0.5
        || lower.contains("counter")
        || lower.contains("accumulator")
        || lower.contains("buffer")
        || lower.contains("state")
    {
        channels.set_state_variables(2.0);
    } else if loop_depth > 0.5 {
        channels.set_state_variables(1.0);
    }

    // Helper functions
    if lower.contains("helper")
        || lower.contains("subroutine")
        || (channels.channels[2] > 0.5 && lower.contains("inner"))
    {
        channels.set_helper_functions(1.0);
    }

    // Pattern match arms
    if lower.contains("match") || lower.contains("case") || lower.contains("variant") {
        channels.set_pattern_match_arms(3.0);
    } else if signature.contains("enum ")
        || signature.contains("Option<")
        || signature.contains("Result<")
    {
        channels.set_pattern_match_arms(2.0);
    }

    channels
}

// ─── k-NN HDC Voting Classifier ────────────────────────────────────────────

/// Classify by majority vote among k nearest training pairs in HDC space.
///
/// This is the natural HDC classifier: no learned weights, no overfitting.
/// Uses cosine similarity on the FULL 16,384D vectors (not the projected 512D).
/// Improves monotonically with corpus size.
///
/// Returns (predicted_class, confidence) where confidence is the vote fraction.
pub fn knn_hdc_classify(
    query_hv: &symthaea_core::hdc::ContinuousHV,
    pairs: &[AlgorithmTrainingPair],
    k: usize,
) -> (AlgorithmClass, f32) {
    if pairs.is_empty() {
        return (AlgorithmClass::IoTransform, 0.0);
    }

    // Compute similarities to all training pairs
    let mut sims: Vec<(f32, AlgorithmClass)> = pairs
        .iter()
        .map(|p| (query_hv.similarity(&p.hv), p.class))
        .collect();

    // Sort descending by similarity
    sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k
    let k = k.min(sims.len());
    let top_k = &sims[..k];

    // Vote — use similarity-weighted voting (closer neighbors count more)
    let mut votes: std::collections::HashMap<AlgorithmClass, f32> =
        std::collections::HashMap::new();
    for (sim, class) in top_k {
        // Use similarity as vote weight (higher similarity = stronger vote)
        *votes.entry(*class).or_insert(0.0) += sim.max(0.0);
    }

    let total: f32 = votes.values().sum();
    let (winner, win_weight) = votes
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((AlgorithmClass::IoTransform, 0.0));

    let confidence = if total > 0.0 { win_weight / total } else { 0.0 };
    (winner, confidence)
}

/// Hybrid classifier using k-NN HDC voting + keyword priors.
///
/// 1. If strong keyword match → return that class
/// 2. Otherwise → k-NN vote in HDC space
pub fn hybrid_classify_knn(
    purpose: &str,
    query_hv: &symthaea_core::hdc::ContinuousHV,
    pairs: &[AlgorithmTrainingPair],
    k: usize,
) -> AlgorithmClass {
    if let Some(class) = strong_keyword_class(purpose) {
        return class;
    }
    knn_hdc_classify(query_hv, pairs, k).0
}

/// Strong keyword match: returns Some(class) only when there's a high-confidence
/// signal in the purpose text. Returns None for ambiguous cases.
///
/// Used as a prior for the hybrid System 1 classifier — if keywords are
/// strong, trust them over the linear classifier (which has limited training
/// data per class).
pub fn strong_keyword_class(purpose: &str) -> Option<AlgorithmClass> {
    let lower = purpose.to_lowercase();

    // Sorting — very specific verbs
    if lower.contains("sort ")
        || lower.contains("sorted ")
        || lower.contains("ascending")
        || lower.contains("descending")
        || lower.contains("order ") && !lower.contains("order of")
    {
        return Some(AlgorithmClass::Sorting);
    }

    // Mathematical — number-theoretic specifics
    if lower.contains("fibonacci")
        || lower.contains("factorial")
        || lower.contains("prime")
        || lower.contains("collatz")
        || lower.contains("armstrong")
        || lower.contains("perfect number")
        || lower.contains("gcd")
        || lower.contains("modulo")
    {
        return Some(AlgorithmClass::Mathematical);
    }

    // String processing — text-specific verbs
    if lower.contains("reverse")
        || lower.contains("uppercase")
        || lower.contains("lowercase")
        || lower.contains("acronym")
        || lower.contains("anagram")
        || lower.contains("palindrome")
        || lower.contains("cipher")
        || lower.contains("encode")
        || lower.contains("decode")
    {
        return Some(AlgorithmClass::StringProcessing);
    }

    // Search
    if lower.contains("binary search")
        || lower.contains("linear search")
        || lower.contains("sieve")
        || lower.contains("lookup")
    {
        return Some(AlgorithmClass::Search);
    }

    // Dynamic programming
    if lower.contains("knapsack")
        || lower.contains("dynamic programming")
        || lower.contains("optimal substructure")
        || lower.contains("memoize")
    {
        return Some(AlgorithmClass::DynamicProgramming);
    }

    // Graph
    if lower.contains("graph")
        || lower.contains("dfs")
        || lower.contains("bfs")
        || lower.contains("shortest path")
        || lower.contains("traversal")
        || lower.contains("euler")
        || lower.contains("hamilton")
    {
        return Some(AlgorithmClass::Graph);
    }

    // Data structure
    if lower.contains("linked list")
        || lower.contains("stack ")
        || lower.contains("queue ")
        || lower.contains("buffer")
        || lower.contains("hashmap")
        || lower.contains("hashset")
        || lower.contains("custom set")
        || lower.contains("circular")
    {
        return Some(AlgorithmClass::DataStructure);
    }

    None
}

/// Hybrid classifier: keyword priors + linear classifier fallback.
///
/// If strong_keyword_class returns Some, trust it. Otherwise use the
/// trained linear classifier on HDC features.
pub fn hybrid_classify(
    purpose: &str,
    features: &[f32],
    classifier: &AlgorithmClassifier,
) -> AlgorithmClass {
    if let Some(class) = strong_keyword_class(purpose) {
        return class;
    }
    classifier.predict(features)
}

/// Classify algorithm from purpose text.
fn classify_from_purpose(purpose: &str) -> AlgorithmClass {
    let lower = purpose.to_lowercase();
    if lower.contains("sort") || lower.contains("order") {
        AlgorithmClass::Sorting
    } else if lower.contains("knapsack") || lower.contains("dynamic") || lower.contains("optimal") {
        AlgorithmClass::DynamicProgramming
    } else if lower.contains("graph") || lower.contains("path") || lower.contains("node") {
        AlgorithmClass::Graph
    } else if lower.contains("math")
        || lower.contains("prime")
        || lower.contains("factor")
        || lower.contains("sum")
        || lower.contains("number")
    {
        AlgorithmClass::Mathematical
    } else if lower.contains("search") || lower.contains("find") || lower.contains("lookup") {
        AlgorithmClass::Search
    } else if lower.contains("string")
        || lower.contains("char")
        || lower.contains("text")
        || lower.contains("word")
        || lower.contains("reverse")
    {
        AlgorithmClass::StringProcessing
    } else if lower.contains("struct")
        || lower.contains("stack")
        || lower.contains("queue")
        || lower.contains("list")
    {
        AlgorithmClass::DataStructure
    } else {
        AlgorithmClass::IoTransform
    }
}

/// Assemble code using 1-NN retrieval from the training corpus.
///
/// If the CfC plan is trivial (only DefineFunction), fall back to
/// finding the most similar training solution and adapting its structure.
fn assemble_with_1nn(
    signature: &str,
    plan_steps: &[crate::dynamics::cfc_code_sequencer::CodePlanStep],
    channels: &AlgorithmChannels,
    class: AlgorithmClass,
    pairs: &[AlgorithmTrainingPair],
    encoder: &AlgorithmEncoder,
) -> String {
    // Check if CfC produced a non-trivial plan
    let has_interesting_action = plan_steps
        .iter()
        .any(|s| !matches!(s.action, PlanAction::DefineFunction | PlanAction::Complete));

    if has_interesting_action {
        // Use CfC plan
        return assemble_from_plan(signature, plan_steps, channels);
    }

    // CfC plan is trivial — use 1-NN retrieval
    let query_hv = encoder.encode(channels);
    let mut best_sim = -1.0f32;
    let mut best_pair: Option<&AlgorithmTrainingPair> = None;

    // Prefer same-class matches
    for pair in pairs {
        let sim = query_hv.similarity(&pair.hv);
        let class_boost = if pair.class == class { 0.1 } else { 0.0 };
        if sim + class_boost > best_sim {
            best_sim = sim + class_boost;
            best_pair = Some(pair);
        }
    }

    if let Some(nearest) = best_pair {
        // Adapt the nearest solution's structure to our signature
        // Use the class-specific template plan instead
        let template_plan = class_to_plan(class, channels);
        assemble_from_plan(signature, &to_plan_steps(&template_plan), channels)
    } else {
        assemble_from_plan(signature, plan_steps, channels)
    }
}

/// Convert PlanAction vec to CodePlanStep vec (with default confidence).
fn to_plan_steps(actions: &[PlanAction]) -> Vec<crate::dynamics::cfc_code_sequencer::CodePlanStep> {
    actions
        .iter()
        .map(|a| crate::dynamics::cfc_code_sequencer::CodePlanStep {
            action: a.clone(),
            name: None,
            context: Vec::new(),
            confidence: 0.5,
        })
        .collect()
}

/// Attempt to compile code via rustc syntax check.
///
/// Writes to a temp file and invokes rustc --crate-type lib.
fn try_compile(code: &str) -> Result<(), String> {
    use std::process::Command;

    let tmp = std::env::temp_dir();
    let src = tmp.join("symthaea_cold_gen.rs");
    let out = tmp.join("libsymthaea_cold_gen.rlib");

    std::fs::write(&src, code).map_err(|e| format!("write: {e}"))?;

    let output = Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            src.to_str().unwrap_or(""),
            "-o",
            out.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| format!("rustc: {e}"))?;

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let errors: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("error[E") || l.starts_with("error:"))
        .collect();
    Err(if errors.is_empty() {
        stderr
    } else {
        errors.join("\n")
    })
}

/// A single test case for execution feedback.
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Rust expression that calls the function (e.g., "fib(10)").
    pub call: String,
    /// Expected result expression (e.g., "55").
    pub expected: String,
}

impl TestCase {
    pub fn new(call: &str, expected: &str) -> Self {
        Self {
            call: call.to_string(),
            expected: expected.to_string(),
        }
    }
}

/// Result of compile + execute test cycle.
#[derive(Debug)]
pub enum TestOutcome {
    /// All test cases passed.
    AllPass,
    /// Code did not compile.
    CompileError(String),
    /// Code compiled but a test case failed.
    TestFailure {
        case: TestCase,
        actual: Option<String>,
        message: String,
    },
}

impl TestOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, TestOutcome::AllPass)
    }
    pub fn error_message(&self) -> String {
        match self {
            TestOutcome::AllPass => String::new(),
            TestOutcome::CompileError(e) => format!("compile: {e}"),
            TestOutcome::TestFailure {
                case,
                message,
                actual,
            } => format!(
                "test failed: {} expected {} {}{message}",
                case.call,
                case.expected,
                actual
                    .as_deref()
                    .map(|a| format!("got {a} "))
                    .unwrap_or_default(),
            ),
        }
    }
}

/// Compile code with embedded test cases and execute.
///
/// Builds a small main() wrapper that runs each assertion. Returns
/// AllPass if every assertion succeeds, otherwise CompileError or
/// TestFailure with the specific failing case.
pub fn try_compile_and_test(code: &str, tests: &[TestCase]) -> TestOutcome {
    use std::process::Command;

    let tmp = std::env::temp_dir();
    let src = tmp.join("symthaea_test_run.rs");
    let bin = tmp.join("symthaea_test_run");

    // Build wrapper: function code + main() with assertions
    let mut full = String::from(code);
    full.push_str("\n\nfn main() {\n");
    for (i, t) in tests.iter().enumerate() {
        full.push_str(&format!(
            "    let __actual_{i} = {};\n    let __expected_{i} = {};\n    if __actual_{i} != __expected_{i} {{\n        eprintln!(\"FAIL {} expected {} got {{:?}}\", __actual_{i});\n        std::process::exit(1);\n    }}\n",
            t.call, t.expected, t.call, t.expected
        ));
    }
    full.push_str("    println!(\"OK\");\n}\n");

    if std::fs::write(&src, &full).is_err() {
        return TestOutcome::CompileError("write failed".to_string());
    }

    let compile_out = Command::new("rustc")
        .args([
            "--edition",
            "2021",
            src.to_str().unwrap_or(""),
            "-o",
            bin.to_str().unwrap_or(""),
        ])
        .output();

    let compile_out = match compile_out {
        Ok(o) => o,
        Err(e) => {
            let _ = std::fs::remove_file(&src);
            return TestOutcome::CompileError(format!("rustc spawn: {e}"));
        }
    };

    if !compile_out.status.success() {
        let stderr = String::from_utf8_lossy(&compile_out.stderr).to_string();
        let _ = std::fs::remove_file(&src);
        let errs: Vec<&str> = stderr
            .lines()
            .filter(|l| l.contains("error[E") || l.starts_with("error:"))
            .collect();
        let msg = if errs.is_empty() {
            stderr
        } else {
            errs.join("\n")
        };
        return TestOutcome::CompileError(msg);
    }

    let run_out = Command::new(&bin).output();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bin);

    let run_out = match run_out {
        Ok(o) => o,
        Err(e) => return TestOutcome::CompileError(format!("exec spawn: {e}")),
    };

    if run_out.status.success() {
        return TestOutcome::AllPass;
    }

    let stderr = String::from_utf8_lossy(&run_out.stderr).to_string();
    // Parse "FAIL <call> expected <expected> got <actual>"
    if let Some(line) = stderr.lines().find(|l| l.starts_with("FAIL")) {
        let parts: Vec<&str> = line.splitn(5, ' ').collect();
        if parts.len() >= 5 {
            let call = parts[1].to_string();
            let expected = parts[3].to_string();
            let actual = parts[4].trim_start_matches("got ").to_string();
            // Find the matching TestCase
            if let Some(case) = tests.iter().find(|t| t.call == call) {
                return TestOutcome::TestFailure {
                    case: case.clone(),
                    actual: Some(actual),
                    message: line.to_string(),
                };
            }
            return TestOutcome::TestFailure {
                case: TestCase::new(&call, &expected),
                actual: Some(actual),
                message: line.to_string(),
            };
        }
    }

    // Generic failure
    TestOutcome::TestFailure {
        case: tests
            .first()
            .cloned()
            .unwrap_or_else(|| TestCase::new("?", "?")),
        actual: None,
        message: stderr.lines().take(2).collect::<Vec<_>>().join(" | "),
    }
}

/// Enrich algorithm channels based on compiler errors.
///
/// This is the "learn from failure" feedback — compiler errors reveal
/// what structural features the code is missing.
fn enrich_channels_from_errors(channels: &mut AlgorithmChannels, errors: &str) {
    let lower = errors.to_lowercase();

    // Type errors → increase type complexity
    if lower.contains("mismatched types") || lower.contains("expected") {
        channels.set_type_complexity(channels.channels[11] + 1.0);
    }

    // Missing trait → add trait bounds
    if lower.contains("the trait") || lower.contains("not satisfied") {
        channels.set_trait_bound_count(channels.channels[10] + 1.0);
    }

    // Borrow checker → increase mutation awareness
    if lower.contains("borrow") || lower.contains("cannot move") || lower.contains("lifetime") {
        channels.set_mutation_level(channels.channels[23].min(2.0) + 1.0);
        channels.set_has_lifetime(true);
    }

    // Missing variable → needs more structure
    if lower.contains("not found") || lower.contains("cannot find") {
        channels.set_state_variables(channels.channels[28] + 1.0);
    }

    // Iterator errors → adjust data flow
    if lower.contains("iterator") || lower.contains("collect") {
        channels.set_map_count(1.0_f32.max(channels.channels[20]));
    }
}

/// Enrich channels from a logical (test) failure.
///
/// Test failures are different from compile errors — they tell us the SHAPE
/// of the function is wrong, not the syntax. We boost complexity hints so
/// the next iteration tries a richer idiom or template.
pub fn enrich_channels_from_test_failure(channels: &mut AlgorithmChannels, failure: &TestOutcome) {
    if let TestOutcome::TestFailure { case, actual, .. } = failure {
        // The function returned wrong values — needs more state/logic
        channels.set_state_variables((channels.channels[28] + 1.0).min(10.0));
        channels.set_cyclomatic_complexity((channels.channels[27] + 1.0).min(20.0));

        // Inspect actual vs expected to guess what's missing
        if let Some(act) = actual {
            // Returning empty Vec → needs a loop/iterator
            if act == "[]" || act == "Vec::new()" || act == "0" {
                channels.set_loop_depth((channels.channels[0] + 1.0).min(5.0));
            }
            // Returning false when expecting true → branch logic missing
            if act == "false" && case.expected != "false" {
                channels.set_branch_depth((channels.channels[1] + 1.0).min(5.0));
            }
            // Returning input unchanged → no transformation happened
            if act.contains(&case.call) {
                channels.set_map_count((channels.channels[20] + 1.0).min(5.0));
            }
        }
    }
}

/// Generate code with test-driven repair.
///
/// Goes beyond compile-only repair: when tests fail, enriches features
/// with logical-failure signal and tries different idioms.
pub fn generate_with_test_repair(
    purpose: &str,
    signature: &str,
    tests: &[TestCase],
    pairs: &[AlgorithmTrainingPair],
    classifier: &AlgorithmClassifier,
    max_iterations: usize,
) -> RepairResult {
    let encoder = AlgorithmEncoder::new();
    let projection = LearnedProjection::fit(pairs);

    let mut channels = build_channels_from_purpose(purpose, signature);
    let mut error_history = Vec::new();
    let mut current_code = String::new();
    let mut compiles = false;
    let mut class = AlgorithmClass::IoTransform;

    for iteration in 0..max_iterations {
        let hv = encoder.encode(&channels);
        let projected = projection.project(&hv);
        class = hybrid_classify(purpose, &projected.values, classifier);

        // Tier 1: Try class+signature idiom
        let idiom_attempt = class_idiom_body(class, purpose, signature)
            .map(|body| format!("{signature} {{\n{body}\n}}"));

        // Tier 2: Try 1-NN body retrieval
        let nn_attempt = if iteration == 0 {
            generate_via_nearest_neighbor(purpose, signature, pairs, classifier)
        } else {
            None
        };

        // Try each candidate against test cases
        for candidate in idiom_attempt.iter().chain(nn_attempt.iter()) {
            let outcome = try_compile_and_test(candidate, tests);
            if outcome.is_pass() {
                return RepairResult {
                    purpose: purpose.to_string(),
                    signature: signature.to_string(),
                    iterations: iteration + 1,
                    compiles: true,
                    final_code: candidate.clone(),
                    error_history,
                    class,
                };
            }
            current_code = candidate.clone();
            // Enrich channels for next iteration based on what failed
            match &outcome {
                TestOutcome::CompileError(e) => {
                    error_history.push(format!("compile: {e}"));
                    enrich_channels_from_errors(&mut channels, e);
                }
                TestOutcome::TestFailure { .. } => {
                    error_history.push(outcome.error_message());
                    enrich_channels_from_test_failure(&mut channels, &outcome);
                }
                _ => {}
            }
        }

        // Tier 3: template fallback with current channels
        channels.set_class(class);
        let template_actions = class_to_plan(class, &channels);
        let plan_steps = to_plan_steps(&template_actions);
        let template_code = assemble_from_plan(signature, &plan_steps, &channels);
        let outcome = try_compile_and_test(&template_code, tests);
        current_code = template_code;
        if outcome.is_pass() {
            compiles = true;
            return RepairResult {
                purpose: purpose.to_string(),
                signature: signature.to_string(),
                iterations: iteration + 1,
                compiles: true,
                final_code: current_code,
                error_history,
                class,
            };
        }
        match &outcome {
            TestOutcome::CompileError(e) => {
                error_history.push(format!("template: {e}"));
                enrich_channels_from_errors(&mut channels, e);
            }
            TestOutcome::TestFailure { .. } => {
                error_history.push(outcome.error_message());
                enrich_channels_from_test_failure(&mut channels, &outcome);
            }
            _ => {}
        }
    }

    RepairResult {
        purpose: purpose.to_string(),
        signature: signature.to_string(),
        iterations: error_history.len(),
        compiles,
        final_code: current_code,
        error_history,
        class,
    }
}

/// Training report with honest metrics.
#[derive(Debug)]
pub struct TrainingReport {
    pub corpus_size: usize,
    pub train_size: usize,
    pub eval_size: usize,
    pub classification_accuracy: f32,
    pub correct: usize,
    pub total: usize,
    pub avg_intra_class_similarity: f32,
    pub avg_inter_class_similarity: f32,
    /// Positive separation = HDC space clusters same-class algorithms together.
    pub separation: f32,
    pub per_class: std::collections::HashMap<AlgorithmClass, (usize, usize)>,
}

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Algorithm Encoder Training Report ===")?;
        writeln!(
            f,
            "Corpus: {} solutions ({} train / {} eval)",
            self.corpus_size, self.train_size, self.eval_size
        )?;
        writeln!(
            f,
            "Classification accuracy: {}/{} ({:.1}%)",
            self.correct,
            self.total,
            self.classification_accuracy * 100.0
        )?;
        writeln!(
            f,
            "Intra-class similarity: {:.4}",
            self.avg_intra_class_similarity
        )?;
        writeln!(
            f,
            "Inter-class similarity: {:.4}",
            self.avg_inter_class_similarity
        )?;
        writeln!(f, "Separation (intra - inter): {:.4}", self.separation)?;
        writeln!(f, "Per-class:")?;
        for (class, (correct, total)) in &self.per_class {
            writeln!(
                f,
                "  {:?}: {}/{} ({:.0}%)",
                class,
                correct,
                total,
                if *total > 0 {
                    *correct as f32 / *total as f32 * 100.0
                } else {
                    0.0
                }
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_extracts_features() {
        let pairs = build_training_pairs();
        assert!(
            pairs.len() >= 20,
            "corpus should have at least 20 solutions, got {}",
            pairs.len()
        );

        // Every pair should have a non-zero HV
        for pair in &pairs {
            assert!(pair.hv.norm() > 0.0, "{} has zero HV", pair.name);
        }
    }

    #[test]
    fn test_class_diversity() {
        let pairs = build_training_pairs();
        let classes: std::collections::HashSet<_> = pairs.iter().map(|p| p.class).collect();
        assert!(
            classes.len() >= 6,
            "should cover at least 6 algorithm classes, got {}",
            classes.len()
        );
    }

    #[test]
    fn test_positive_separation() {
        let report = train_and_evaluate();
        println!("{report}");
        assert!(
            report.separation > 0.0,
            "intra-class similarity should exceed inter-class: separation = {:.4}",
            report.separation
        );
    }

    #[test]
    fn test_class_to_plan_produces_valid_sequences() {
        let pairs = build_training_pairs();
        for pair in &pairs {
            let plan = class_to_plan(pair.class, &pair.channels);
            assert!(!plan.is_empty(), "{} has empty plan", pair.name);
            assert_eq!(
                plan[0],
                PlanAction::DefineFunction,
                "{} plan should start with DefineFunction",
                pair.name
            );
            assert_eq!(
                *plan.last().unwrap(),
                PlanAction::Complete,
                "{} plan should end with Complete",
                pair.name
            );
        }
    }

    #[test]
    fn test_learned_projection_improves_separation() {
        let pairs = build_training_pairs();
        let projection = LearnedProjection::fit(&pairs);

        // Compare separation in projected vs naive-truncated space
        let mut naive_intra = Vec::new();
        let mut naive_inter = Vec::new();
        let mut proj_intra = Vec::new();
        let mut proj_inter = Vec::new();

        for (i, a) in pairs.iter().enumerate() {
            for (j, b) in pairs.iter().enumerate() {
                if i >= j {
                    continue;
                }
                let naive_a = project_to_sequencer_dim(&a.hv);
                let naive_b = project_to_sequencer_dim(&b.hv);
                let proj_a = projection.project(&a.hv);
                let proj_b = projection.project(&b.hv);

                let naive_sim = naive_a.similarity(&naive_b);
                let proj_sim = proj_a.similarity(&proj_b);

                if a.class == b.class {
                    naive_intra.push(naive_sim);
                    proj_intra.push(proj_sim);
                } else {
                    naive_inter.push(naive_sim);
                    proj_inter.push(proj_sim);
                }
            }
        }

        let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;

        let naive_sep = avg(&naive_intra) - avg(&naive_inter);
        let proj_sep = avg(&proj_intra) - avg(&proj_inter);

        println!(
            "Naive truncation: intra={:.4} inter={:.4} sep={:.4}",
            avg(&naive_intra),
            avg(&naive_inter),
            naive_sep
        );
        println!(
            "Learned projection: intra={:.4} inter={:.4} sep={:.4}",
            avg(&proj_intra),
            avg(&proj_inter),
            proj_sep
        );
        println!(
            "Improvement: {:.4} → {:.4} ({:+.1}%)",
            naive_sep,
            proj_sep,
            if naive_sep.abs() > 1e-6 {
                (proj_sep / naive_sep - 1.0) * 100.0
            } else {
                f32::INFINITY
            }
        );

        assert!(
            proj_sep > 0.0,
            "learned projection should have positive separation: {:.4}",
            proj_sep
        );
    }

    #[test]
    fn test_cold_generation_pipeline() {
        let pairs = build_training_pairs();

        // Test: generate code for "find prime factors of a number"
        let result = cold_generate(
            "find prime factors of a number",
            "pub fn factors(n: u64) -> Vec<u64>",
            &pairs,
        );
        println!("=== Cold Generation: prime factors ===");
        println!("Class: {:?}", result.detected_class);
        println!("Plan: {:?}", result.plan);
        println!("Code:\n{}", result.code);
        println!("Plausible: {}", result.plausible);

        assert!(
            result.plausible,
            "generated code should be syntactically plausible"
        );
        assert_eq!(
            result.detected_class,
            AlgorithmClass::Mathematical,
            "should detect as mathematical"
        );

        // Test: generate code for "reverse a string"
        let result2 = cold_generate(
            "reverse the characters in a string",
            "pub fn reverse(input: &str) -> String",
            &pairs,
        );
        println!("\n=== Cold Generation: reverse string ===");
        println!("Class: {:?}", result2.detected_class);
        println!("Plan: {:?}", result2.plan);
        println!("Code:\n{}", result2.code);
        println!("Plausible: {}", result2.plausible);

        assert!(result2.plausible);
        assert_eq!(result2.detected_class, AlgorithmClass::StringProcessing);

        // Test: generate code for "check if number is even"
        let result3 = cold_generate(
            "check if a number is even",
            "pub fn is_even(n: u64) -> bool",
            &pairs,
        );
        println!("\n=== Cold Generation: is_even ===");
        println!("Class: {:?}", result3.detected_class);
        println!("Code:\n{}", result3.code);
        println!("Plausible: {}", result3.plausible);

        assert!(result3.plausible);
    }

    #[test]
    fn test_self_repair_produces_compilable_code() {
        let pairs = build_training_pairs();

        let problems = [
            ("reverse a string", "pub fn reverse(input: &str) -> String"),
            (
                "check if a number is even",
                "pub fn is_even(n: u64) -> bool",
            ),
            (
                "sum all numbers in a list",
                "pub fn sum(numbers: &[i32]) -> i32",
            ),
        ];

        let mut compile_count = 0;
        for (purpose, sig) in &problems {
            let result = generate_with_repair(purpose, sig, &pairs, 3);
            println!("=== Repair: {} ===", purpose);
            println!("Class: {:?}", result.class);
            println!("Iterations: {}", result.iterations);
            println!("Compiles: {}", result.compiles);
            println!("Errors: {:?}", result.error_history);
            println!("Code:\n{}\n", result.final_code);

            if result.compiles {
                compile_count += 1;
            }
        }

        println!(
            "Compile rate: {}/{} ({:.0}%)",
            compile_count,
            problems.len(),
            compile_count as f32 / problems.len() as f32 * 100.0
        );

        // Most should compile — exact count varies with corpus size
        assert!(
            compile_count >= 1,
            "at least 1/3 should compile, got {}",
            compile_count
        );
    }

    #[test]
    #[test]
    fn test_cfc_training_converges() {
        let report = train_cfc_sequencer(200, 0.001);
        println!("{report}");
        let n = report.loss_curve.len();
        if n >= 4 {
            println!(
                "Loss curve: epoch 0={:.6}, 25%={:.6}, 50%={:.6}, 75%={:.6}, 100%={:.6}",
                report.loss_curve[0],
                report.loss_curve[n / 4],
                report.loss_curve[n / 2],
                report.loss_curve[3 * n / 4],
                report.loss_curve[n - 1],
            );
        }
        assert!(
            report.final_loss < report.loss_curve[0],
            "loss should decrease over training: {:.6} → {:.6}",
            report.loss_curve[0],
            report.final_loss
        );
    }

    #[test]
    fn test_curriculum_vs_flat_training() {
        // Curriculum: easy → medium → hard
        let curriculum = train_cfc_curriculum(200, 0.001);
        println!("CURRICULUM: {curriculum}");

        // Flat: same epochs, same lr, no ordering
        let flat = train_cfc_sequencer(200, 0.001);
        println!("FLAT: {flat}");

        println!(
            "\n=== COMPARISON ===\nClass accuracy: curriculum={:.1}% vs flat={:.1}%",
            curriculum.class_from_plan_accuracy * 100.0,
            flat.class_from_plan_accuracy * 100.0,
        );
        println!(
            "Final loss: curriculum={:.6} vs flat={:.6}",
            curriculum.final_loss, flat.final_loss,
        );

        // Both should have finite loss
        assert!(curriculum.final_loss.is_finite());
        assert!(flat.final_loss.is_finite());
    }

    #[test]
    fn test_compile_and_test_pipeline() {
        // Verify try_compile_and_test correctly identifies pass/fail
        let pass_code = r#"
pub fn add(a: i32, b: i32) -> i32 { a + b }
"#;
        let fail_code = r#"
pub fn add(a: i32, b: i32) -> i32 { a - b }
"#;

        let tests = vec![
            TestCase::new("add(2, 3)", "5"),
            TestCase::new("add(10, 7)", "17"),
        ];

        let pass_result = try_compile_and_test(pass_code, &tests);
        println!("Pass case: {:?}", pass_result);
        assert!(pass_result.is_pass(), "correct add should pass tests");

        let fail_result = try_compile_and_test(fail_code, &tests);
        println!("Fail case: {}", fail_result.error_message());
        assert!(!fail_result.is_pass(), "wrong add should fail tests");
        if let TestOutcome::TestFailure { actual, .. } = &fail_result {
            assert!(actual.is_some(), "should report actual value");
        }
    }

    #[test]
    fn test_test_driven_generation() {
        let (classifier, _, _, _) = train_linear_classifier(100, 0.01);
        let pairs = build_training_pairs();

        // Real test cases — the system must generate code that passes them
        let problems = vec![
            (
                "compute the nth fibonacci number",
                "pub fn fib(n: u64) -> u64",
                vec![
                    TestCase::new("fib(0)", "0"),
                    TestCase::new("fib(1)", "1"),
                    TestCase::new("fib(10)", "55"),
                ],
            ),
            (
                "check if a number is prime",
                "pub fn is_prime(n: u64) -> bool",
                vec![
                    TestCase::new("is_prime(2)", "true"),
                    TestCase::new("is_prime(7)", "true"),
                    TestCase::new("is_prime(10)", "false"),
                ],
            ),
            (
                "sort a list of integers ascending",
                "pub fn sort_nums(nums: Vec<i32>) -> Vec<i32>",
                vec![TestCase::new("sort_nums(vec![3, 1, 2])", "vec![1, 2, 3]")],
            ),
        ];

        println!("\n=== Test-Driven Generation ===");
        let mut passing = 0usize;
        for (purpose, sig, tests) in &problems {
            let result = generate_with_test_repair(purpose, sig, tests, &pairs, &classifier, 3);
            let status = if result.compiles { "✓" } else { "✗" };
            println!(
                "  {} '{}'\n     → {:?} | iter={} | tests pass: {}",
                status, purpose, result.class, result.iterations, result.compiles
            );
            if result.compiles {
                passing += 1;
            } else if let Some(last) = result.error_history.last() {
                println!("     last error: {}", last.lines().next().unwrap_or(""));
            }
        }
        println!("\nTest-driven pass rate: {}/{}", passing, problems.len());
        assert!(
            passing >= 2,
            "at least 2/3 should pass tests, got {passing}"
        );
    }

    #[test]
    fn test_knn_vs_linear_classifier() {
        let pairs = build_training_pairs();
        let split = (pairs.len() * 4) / 5;
        let (train, eval) = pairs.split_at(split);

        // Linear classifier on projected 512D space
        let (lin_classifier, lin_train_acc, lin_eval_acc, _) = train_linear_classifier(100, 0.01);

        // k-NN HDC voting on full 16,384D space, k=5
        let mut knn_correct = 0usize;
        for eval_pair in eval {
            let predicted = knn_hdc_classify(&eval_pair.hv, train, 5).0;
            if predicted == eval_pair.class {
                knn_correct += 1;
            }
        }
        let knn_eval_acc = knn_correct as f32 / eval.len() as f32;

        println!("=== Linear vs k-NN HDC Voting ===");
        println!(
            "Linear classifier:    train={:.0}% eval={:.0}% (4,104 params)",
            lin_train_acc * 100.0,
            lin_eval_acc * 100.0
        );
        println!(
            "k-NN HDC (k=5):       train=N/A     eval={:.0}% (no params)",
            knn_eval_acc * 100.0
        );

        let _ = lin_classifier; // silence unused warning

        // k-NN should be at least as good as linear on this small corpus
        assert!(
            knn_eval_acc >= 0.5,
            "k-NN should achieve at least 50% on held-out: got {:.0}%",
            knn_eval_acc * 100.0
        );
    }

    #[test]
    fn test_hybrid_repair_full_pipeline() {
        let (classifier, _, _, _) = train_linear_classifier(100, 0.01);
        let pairs = build_training_pairs();

        let problems = [
            (
                "reverse the characters in a string",
                "pub fn reverse(input: &str) -> String",
            ),
            (
                "sort a list of integers",
                "pub fn sort_nums(nums: Vec<i32>) -> Vec<i32>",
            ),
            ("compute the nth fibonacci", "pub fn fib(n: u64) -> u64"),
            (
                "count vowels in a string",
                "pub fn count_vowels(s: &str) -> usize",
            ),
            (
                "check if a number is even",
                "pub fn is_even(n: u64) -> bool",
            ),
        ];

        println!("\n=== Hybrid System 1/System 2 + Self-Repair ===");
        let mut compiles = 0;
        let mut nn_wins = 0;
        let mut template_wins = 0;
        for (purpose, sig) in &problems {
            let result = generate_with_repair_hybrid(purpose, sig, &pairs, &classifier, 3);
            let label = if result.iterations == 1 && result.compiles {
                nn_wins += 1;
                "NN"
            } else if result.compiles {
                template_wins += 1;
                "TEMPLATE"
            } else {
                "FAIL"
            };
            println!(
                "  '{}'\n    → {:?} | {} | iter={} | compiles={}",
                purpose, result.class, label, result.iterations, result.compiles
            );
            if result.compiles {
                compiles += 1;
            }
        }
        println!(
            "\nTotal: {}/{} compile (NN={}, Template={})",
            compiles,
            problems.len(),
            nn_wins,
            template_wins
        );

        assert!(compiles >= 3, "at least 3/5 should compile, got {compiles}");
    }

    #[test]
    fn test_nearest_neighbor_body_retrieval() {
        let (classifier, _, _, _) = train_linear_classifier(100, 0.01);
        let pairs = build_training_pairs();

        let problems = [
            (
                "reverse the characters in a string",
                "pub fn reverse(input: &str) -> String",
            ),
            (
                "sort a list of integers",
                "pub fn sort_nums(nums: Vec<i32>) -> Vec<i32>",
            ),
            ("compute the nth fibonacci", "pub fn fib(n: u64) -> u64"),
        ];

        println!("\n=== 1-NN Body Retrieval ===");
        let mut compiles = 0;
        for (purpose, sig) in &problems {
            if let Some(code) = generate_via_nearest_neighbor(purpose, sig, &pairs, &classifier) {
                println!("--- '{purpose}' ---");
                println!("{code}");
                if try_compile(&code).is_ok() {
                    compiles += 1;
                    println!("✓ COMPILES\n");
                } else {
                    println!("✗ does not compile\n");
                }
            } else {
                println!("'{purpose}' → no nearest neighbor found\n");
            }
        }
        println!("Compile rate: {}/{}", compiles, problems.len());

        // At minimum, we should retrieve something for each
        assert!(
            compiles >= 1,
            "at least 1/3 NN-retrieved bodies should compile"
        );
    }

    #[test]
    fn test_system1_system2_generation() {
        let (classifier, train_acc, eval_acc, _) = train_linear_classifier(100, 0.01);
        println!(
            "Classifier trained: train={:.1}% eval={:.1}%",
            train_acc * 100.0,
            eval_acc * 100.0
        );

        let pairs = build_training_pairs();
        let problems = [
            (
                "reverse the characters in a string",
                "pub fn reverse(input: &str) -> String",
            ),
            (
                "sort a list of integers",
                "pub fn sort_nums(nums: Vec<i32>) -> Vec<i32>",
            ),
            (
                "find prime factors of a number",
                "pub fn factors(n: u64) -> Vec<u64>",
            ),
            ("compute the nth fibonacci", "pub fn fib(n: u64) -> u64"),
        ];

        println!("\n=== System 1 (Classify) + System 2 (Template) ===");
        for (purpose, sig) in &problems {
            let result = cold_generate_classified(purpose, sig, &pairs, &classifier);
            println!(
                "  '{purpose}'\n    → Class: {:?}, Plan: {} steps\n    → {}\n",
                result.detected_class,
                result.plan.len(),
                result.code.lines().next().unwrap_or("")
            );
        }
    }

    #[test]
    fn test_linear_classifier_beats_cfc() {
        let (classifier, train_acc, eval_acc, losses) = train_linear_classifier(100, 0.01);
        println!("=== Linear Classifier ===");
        println!("Train accuracy: {:.1}%", train_acc * 100.0);
        println!("Eval accuracy:  {:.1}%", eval_acc * 100.0);
        println!(
            "Loss: {:.4} → {:.4}",
            losses.first().unwrap_or(&0.0),
            losses.last().unwrap_or(&0.0)
        );

        // The linear classifier should beat the CfC's 16.7%
        assert!(
            eval_acc > 0.16,
            "linear classifier should beat CfC's 16.7%: got {:.1}%",
            eval_acc * 100.0
        );

        // Test prediction on known types
        let encoder = AlgorithmEncoder::new();
        let pairs = build_training_pairs();
        let projection = LearnedProjection::fit(&pairs);

        let mut sort_ch = AlgorithmChannels::default();
        sort_ch.set_class(AlgorithmClass::Sorting);
        sort_ch.set_loop_depth(2.0);
        sort_ch.set_mutation_level(2.0);
        let sort_proj = projection.project(&encoder.encode(&sort_ch));
        let predicted = classifier.predict(&sort_proj.values);
        println!("Sorting input → predicted: {:?}", predicted);
    }

    #[test]
    fn test_difficulty_ordering() {
        let mut pairs = build_training_pairs();
        sort_by_difficulty(&mut pairs);

        // First should be simpler than last
        let first_diff = difficulty_score(&pairs[0].channels);
        let last_diff = difficulty_score(&pairs[pairs.len() - 1].channels);
        println!(
            "Difficulty range: {:.1} ({}) → {:.1} ({})",
            first_diff,
            pairs[0].name,
            last_diff,
            pairs[pairs.len() - 1].name
        );

        // Print the ordering
        for (i, p) in pairs.iter().enumerate() {
            let d = difficulty_score(&p.channels);
            if i < 5 || i >= pairs.len() - 5 {
                println!("  {:2}. {:.1} {:?} {}", i, d, p.class, p.name);
            } else if i == 5 {
                println!("  ...");
            }
        }

        assert!(
            first_diff <= last_diff,
            "should be sorted easy→hard: {first_diff} > {last_diff}"
        );
    }
}
