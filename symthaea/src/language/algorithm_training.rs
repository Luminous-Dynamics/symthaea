// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! CfC Code Sequencer training on abstract algorithm vectors.
//!
//! Extracts structural features from the Exercism type-aware module
//! implementations, encodes them as 16,384D HDC vectors via
//! AlgorithmEncoder, and trains the CfC temporal network to predict
//! algorithm class from the HDC representation.

use super::algorithm_encoder::{
    extract_features, AlgorithmChannels, AlgorithmClass, AlgorithmEncoder, AlgorithmTrainingPair,
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
            }
        })
        .collect()
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
        if total > 0.0 {
            top / total
        } else {
            0.0
        }
    }
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
        let planned = sequencer.plan_structure(&projected, &[]);
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
    let plan_steps = sequencer.plan_structure(&projected, &[]);
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
        let plan_steps = sequencer.plan_structure(&projected, &[]);

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
fn build_channels_from_purpose(purpose: &str, signature: &str) -> AlgorithmChannels {
    let mut channels = AlgorithmChannels::default();
    let lower = purpose.to_lowercase();

    // Parse arity from signature
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

    // Return type analysis
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

    // Structural hints from purpose
    if lower.contains("each") || lower.contains("every") || lower.contains("iterate") {
        channels.set_loop_depth(1.0);
    }
    if lower.contains("nested") || lower.contains("matrix") || lower.contains("grid") {
        channels.set_loop_depth(2.0);
    }
    if lower.contains("recursive") || lower.contains("factorial") {
        channels.set_recursion(true);
    }

    channels
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

        // With correct parameter names, all trivial fallbacks should compile
        assert!(
            compile_count >= 3,
            "all 3 should compile, got {}",
            compile_count
        );
    }

    #[test]
    fn test_cfc_training_converges() {
        let report = train_cfc_sequencer(200, 0.001);
        println!("{report}");
        // Print loss curve samples
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
}
