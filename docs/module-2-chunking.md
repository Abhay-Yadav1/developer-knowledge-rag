# Module 2: Chunking Strategy

This document explains how the Developer Knowledge & Codebase Assistant splits repository files into chunks before indexing them (the "Clean and chunk" stage of the ingestion pipeline in `module-1-architecture.md`). It compares three strategies, shows how each would split a Java file, and states the initial choice for Java files and Markdown documentation.

## Contents
1. [Why We Split Documents into Chunks](#1-why-we-split-documents-into-chunks)
2. [Chunking Strategies Compared](#2-chunking-strategies-compared)
3. [Worked Example: A Java File with login(), logout(), and resetPassword()](#3-worked-example-a-java-file-with-login-logout-and-resetpassword)
4. [Initial Choice for Java and Markdown](#4-initial-choice-for-java-and-markdown)
5. [When Chunking Separates a Method from Important Context](#5-when-chunking-separates-a-method-from-important-context)

---

## 1. Why We Split Documents into Chunks

We could embed each file as a single unit, but that works poorly for a codebase assistant:

| Reason | Explanation |
|---|---|
| **Embedding quality** | One vector can only represent so much meaning. A vector for a whole 800-line class is a blurry average of everything in it, so a question about one method matches it weakly. A focused chunk gets a sharp vector. |
| **Retrieval precision** | We want to retrieve the 20 lines that answer the question, not an entire file where the answer is buried. |
| **Context window and cost** | The LLM can only read a limited amount of text per request, and every extra token costs time and money. Small, relevant chunks leave room for several sources. |
| **Less noise** | Irrelevant code in the prompt distracts the model and increases the chance of a wrong or mixed-up answer. |
| **Precise citations** | Our citations need a file path, revision, and line range (Module 1, Section 4). Smaller chunks give tight line ranges that a developer can check quickly. |
| **Efficient updates** | When one file changes, only its chunks are replaced (Module 1, Section 3). |

The trade-off is that **chunks that are too small lose context, and chunks that are too large lose precision.** Choosing a strategy is about balancing the two.

---

## 2. Chunking Strategies Compared

### 2.1 Fixed-size chunking
Split the text every *N* characters or tokens (for example 500 tokens), usually with an overlap of a few lines between neighbouring chunks. It ignores what the text means.

- **Pros:** Very simple, fast, works on any file type, predictable chunk sizes.
- **Cons:** Cuts through the middle of methods, sentences, and code blocks. Separates Javadoc from its method. Chunks have no notion of "class" or "section".

### 2.2 Recursive chunking
Try to split on the largest natural separator first (for example blank lines), and only fall back to smaller separators (single newlines, then spaces) when a piece is still larger than the size limit. Small pieces are then merged back together up to the limit. For code, the separator list can be language-aware (class, method, and blank-line boundaries).

- **Pros:** Respects paragraph and block boundaries much more often than fixed-size splitting. Still simple and works on any text.
- **Cons:** Boundaries are guessed from text patterns, not understood. Blank lines inside a method can split it. Several small methods may be merged into one chunk. A very long method still gets cut at line level.

### 2.3 Structure-aware chunking
Parse the file and split along its real structure. For Java, parse the syntax tree (for example with tree-sitter or JavaParser) and make one chunk per method or constructor, plus a chunk for the class header and fields. For Markdown, split by headings and keep code blocks and tables intact. Each chunk carries metadata about where it sits (class, method, heading path).

- **Pros:** Chunks are meaningful units (a whole method, a whole section). Tight, accurate line ranges. Rich metadata for filtering and citations. Context (class name, headings) can be attached to every chunk.
- **Cons:** More code to build and maintain; needs a parser per language. Fails on files that don't parse (broken syntax, templates), so a fallback is needed. Very large methods still need further splitting.

### Summary

| | Fixed-size | Recursive | Structure-aware |
|---|---|---|---|
| Splits on | Size only | Separators, in priority order | Syntax tree / headings |
| Respects method boundaries | No | Sometimes | Yes |
| Keeps Javadoc with its method | Not guaranteed | Usually | Yes |
| Context metadata (class, heading) | None | None | Yes |
| Implementation effort | Lowest | Low | Highest |
| Works on any file type | Yes | Yes | Only with a parser |

---

## 3. Worked Example: A Java File with login(), logout(), and resetPassword()

We use this illustrative `AuthService.java` (line numbers on the left are not part of the file). The class doc, the constant `MAX_FAILED_ATTEMPTS`, and the injected fields are all things the methods depend on.

```text
 1  package com.example.auth;
 2
 3  import java.time.Instant;
 4
 5  /**
 6   * Handles user authentication. Sessions expire after 30 minutes.
 7   */
 8  public class AuthService {
 9
10      private static final int MAX_FAILED_ATTEMPTS = 5;
11      private final UserRepository users;
12      private final SessionStore sessions;
13
14      /** Authenticates a user and starts a session. */
15      public Session login(String username, String password) {
16          User user = users.findByUsername(username);
17          if (user == null || user.isLocked()) {
18              throw new AuthException("Invalid credentials");
19          }
20          if (!PasswordHasher.matches(password, user.getHash())) {
21              user.recordFailedAttempt();
22              if (user.getFailedAttempts() >= MAX_FAILED_ATTEMPTS) {
23                  user.lock();
24              }
25              throw new AuthException("Invalid credentials");
26          }
27          user.clearFailedAttempts();
28          return sessions.create(user.getId(), Instant.now());
29      }
30
31      /** Ends the user's session. */
32      public void logout(String sessionId) {
33          sessions.invalidate(sessionId);
34      }
35
36      /** Resets the password and invalidates all sessions. */
37      public void resetPassword(String username, String newPassword) {
38          User user = users.findByUsername(username);
39          user.setHash(PasswordHasher.hash(newPassword));
40          user.clearFailedAttempts();
41          sessions.invalidateAll(user.getId());
42      }
43  }
```

*The chunk sizes below are chosen to illustrate the behaviour on this small file. Real settings would be larger (see Section 4).*

### 3.1 Fixed-size (about 12 lines per chunk, no overlap)

| Chunk | Lines | Content | Problem |
|---|---|---|---|
| 1 | 1-12 | Package, import, class doc, class declaration, constant, fields | Contains no method at all, so it rarely matches a method question |
| 2 | 13-24 | Start of `login()`: lookup, locked check, and the failed-attempt branch | `login()` is **cut in half**: this part has the lock logic but not the successful-login path |
| 3 | 25-36 | End of `login()`, all of `logout()`, and only the **Javadoc line** of `resetPassword()` | A mix of three things; the `resetPassword` comment is **separated from its method** |
| 4 | 37-43 | Body of `resetPassword()` and the closing brace | The method has lost its description, and nothing says it belongs to `AuthService` |

Adding a small overlap (for example 2 lines) would repeat some neighbouring lines, but `login()` would still be split across chunks.

### 3.2 Recursive (split on blank lines first, merge up to about 20 lines)

The blank lines in this file happen to fall between members, so the first-level split gives these blocks: package line, import, class doc and declaration, fields, `login()` with its Javadoc, `logout()` with its Javadoc, and `resetPassword()` with its Javadoc. Small neighbouring blocks are then merged up to the size limit:

| Chunk | Lines | Content | Notes |
|---|---|---|---|
| 1 | 1-12 | Package, import, class doc, constant, fields | Reasonable "class header" chunk |
| 2 | 14-29 | `login()` with its Javadoc | Whole method kept together |
| 3 | 31-43 | `logout()` and `resetPassword()` together | Two unrelated methods merged because both are small, so a `logout` question also retrieves `resetPassword` |

This result is good, but only because the file is neatly formatted and `login()` fits within the limit. If `login()` contained blank lines inside its body, it could be split in the middle. If it were longer than the limit, the splitter would fall back to cutting at single lines, giving the same problem as fixed-size chunking.

### 3.3 Structure-aware (one chunk per method, plus a class chunk)

The file is parsed and each method becomes its own chunk, with a short context header (path, package, class, class doc) added to the text that is embedded:

| Chunk | Lines | Content |
|---|---|---|
| Class chunk | 1-12 | Package, import, class doc, constant, fields, and a list of method signatures |
| `login` | 14-29 | Javadoc and the complete method |
| `logout` | 31-34 | Javadoc and the complete method |
| `resetPassword` | 36-42 | Javadoc and the complete method |

Example of the text that is embedded for the `login` chunk:

```text
File: src/main/java/com/example/auth/AuthService.java
Package: com.example.auth    Class: AuthService
Class doc: Handles user authentication. Sessions expire after 30 minutes.
Lines: 14-29

/** Authenticates a user and starts a session. */
public Session login(String username, String password) { ... }
```

Each chunk also stores metadata: `file_path`, `class`, `method`, `start_line`, `end_line`, and `source_type: code`. This gives a precise citation such as `AuthService.login`, `L14-L29`, and the retriever can filter by class or method name.

### 3.4 Side-by-side result

| | Fixed-size | Recursive | Structure-aware |
|---|---|---|---|
| `login()` kept whole | No (split across chunks 2 and 3) | Yes (in this file) | Yes |
| `resetPassword()` kept with its Javadoc | No | Yes | Yes |
| `logout()` kept separate | No | No (merged with `resetPassword`) | Yes |
| Class name available to the chunk | No | No | Yes (header and metadata) |
| Exact method-level line range | No | Approximately | Yes |

---

## 4. Initial Choice for Java and Markdown

The values below are **starting points to test and tune**, not final settings.

### 4.1 Java files: structure-aware, with recursive fallback

**Choice:** parse each file and create **one chunk per method or constructor** (with its Javadoc and annotations), plus **one class-level chunk** (package, imports, class doc, constants, fields, method signature list).

**Why:**
- A method is the natural unit that developers ask about ("what does `login` do?").
- It keeps the Javadoc with its method and gives the exact line ranges that our citations require.
- The class and method names become searchable metadata.
- Java has a stable grammar and good parsers, so the extra effort is manageable.

**Rules and fallbacks:**
- Attach a **context header** (path, package, class, class doc) to every chunk before embedding.
- If a method is **larger than the size limit** (for example, about 800 to 1,000 tokens as a first guess), split it recursively at statement or block boundaries and **repeat the method signature** at the top of each part.
- If a file **cannot be parsed**, or is not Java (for example, a config or template file), fall back to **recursive chunking** with code-aware separators.
- Very small methods (such as getters) may be grouped with their neighbours to avoid thousands of tiny chunks.
- Overlap is not needed between method chunks, because the boundaries are real.

### 4.2 Markdown documentation: structure-aware by heading, with recursive splitting inside long sections

**Choice:** split at headings (H1 to H3), and prefix each chunk with its **heading path**, for example `Authentication > Password reset > Steps`.

**Why:**
- Documentation is organised by headings, and each section usually answers one question.
- The heading path restores context that a paragraph like "Run the following command" would otherwise lack.

**Rules and fallbacks:**
- Never split inside a **fenced code block** or a **table**.
- If a section is longer than the limit (a first guess of about 400 to 500 tokens), split it **recursively** by paragraph, repeating the heading path on every part, with a small overlap (about 10 to 15%) between parts.
- Merge very short sections (for example a heading with a single sentence) into their parent or neighbour.
- Store `heading_path` in the metadata and show it in the citation next to the line range.

### 4.3 How we will decide whether it works
Build a small set of test questions with known answers (for example "How many failed logins lock an account?", "What does `resetPassword` do to existing sessions?", "How do I configure the database?"). For each strategy, measure whether the correct chunk appears in the top results and whether the final answer is correct and cited properly. Change the size limits and compare, rather than trusting these first values.

---

## 5. When Chunking Separates a Method from Important Context

A method rarely stands alone. Even a well-chunked method depends on things outside its own lines.

### 5.1 What context can be lost

| Missing context | Example from `AuthService` |
|---|---|
| **Enclosing class, package, file path** | The `login` chunk doesn't say it belongs to `AuthService`, so it can be confused with another class that also has a `login` method |
| **Constants and fields** | `login()` uses `MAX_FAILED_ATTEMPTS`, but its value (5) is defined on line 10, in a different chunk |
| **Class-level documentation** | "Sessions expire after 30 minutes" is only in the class Javadoc |
| **Javadoc and comments** | In fixed-size chunking, the description of `resetPassword` ends up in a different chunk than the method |
| **Imports and types** | Without the import, `Instant` or `PasswordHasher` is unexplained |
| **Called methods and overrides** | What `sessions.invalidateAll()` really does is in another file |
| **Markdown heading** | A step such as "Run the migration" has no meaning without knowing which feature it belongs to |

### 5.2 What can go wrong

1. **Retrieval misses.** The chunk doesn't contain the words from the question (for example "lock" or "attempts"), so it isn't found.
2. **Guessing.** The LLM sees `>= MAX_FAILED_ATTEMPTS` but not the value, and may invent a number (such as 3).
3. **Wrong attribution.** An answer about "the `login` method" is taken from the wrong class or module.
4. **Misleading citations.** The citation shows a line range, but the reader can't tell which class it refers to.
5. **Incomplete answers.** The model describes half of a split method as if it were the whole.

**Example.** Question: "How many failed attempts lock an account?" If only the `login` chunk (L14-L29) is retrieved, the answer depends on `MAX_FAILED_ATTEMPTS`, whose value sits in L10. Without that line, the system must not guess.

### 5.3 How the design reduces the problem

| Mitigation | How it helps |
|---|---|
| **Context header on every chunk** (path, package, class, class doc) | The embedded text carries the "where am I" information, which helps both retrieval and answering |
| **Heading path for Markdown** | Every documentation chunk knows which section it belongs to |
| **Keep Javadoc and annotations with the method** | Avoids detached comments |
| **Class-level chunk with fields, constants, and method list** | Gives the system a place to retrieve constants and an overview of the class |
| **Resolve simple same-file constants** | For constants used in a chunk, add a line like `MAX_FAILED_ATTEMPTS = 5` to its context header |
| **Overlap** (fixed-size, recursive, and long-section splitting) | Repeats a few lines across boundaries so nothing falls exactly on a cut |
| **Neighbour or parent expansion ("small-to-big")** | Retrieve the small chunk, then also fetch the class chunk or the previous and next chunks (stored via `chunk_id` links) before building the prompt |
| **Store structural metadata** | Class, method, and heading are shown in the citation and can be used as filters |
| **Honest behaviour when context is missing** | If a needed symbol or value is not in the retrieved context, the answer says so and cites what it did find, instead of guessing (consistent with the "unsupported answers" rule in Module 1) |

No chunking strategy removes this problem completely, because code depends on other code. The goal is to make sure the **most important context travels with each chunk**, and that the system **admits uncertainty** when it doesn't.