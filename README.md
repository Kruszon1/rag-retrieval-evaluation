# Retrieval Evaluation for a RAG Pipeline

A small project to learn how the retrieval part of a Retrieval-Augmented Generation pipeline works. I index Attention Is All You Need, BERT and Retrieval-Augmented Generation and search them with sentence embeddings and measure how well the search finds the right passage. The main question: how does chunk size affect retrieval quality?

The project covers retrieval only. It returns passages together with their paper and page; it does not generate answers.

**How it works**

1. Download three public PDFs from arXiv when the notebook runs
2. Extract the text page by page
3. Split each page into overlapping chunks (800 characters with 150 overlap for the demo search; the evaluation compares 400, 800 and 1200)
4. Turn each chunk into an embedding
5. Store the vectors in a vector database
6. For a question: embed it and return the closest chunks with paper and page

**Tech stack**

Python, pypdf, sentence-transformers, ChromaDB, pandas, Google Colab

**Run it**

1. Open “Rag-retrieval-evaluation” in Google Colab
2. Run all code cells one after another

**Evaluation**

36 test questions. Each has a known source paper and an exact answer phrase taken from that paper. I compared chunk sizes of 400, 800 and 1200 characters. To keep the comparison fair, the number of retrieved chunks k is chosen so that each chunk size returns about the same amount of text (k = 9, 4 and 3)

- paper@1: the top result comes from the correct paper (easy test)
- Phrase in top k: one of the top k chunks really contains the answer phrase (hard test)
- MRR: mean of 1/rank of the first chunk containing the answer phrase, searched in the top 10

| Chunk size | k | Chunks | paper@1 | Phrase in top k | MRR |
|---|---|---|---|---|---|
| 400 | 9 | 706 | 32/36 | 27/36 | 0.364 |
| 800 | 4 | 285 | 36/36 | 23/36 | 0.465 |
| 1200 | 3 | 184 | 33/36 | 15/36 | 0.428 |

**Conclusion**

- Finding the right paper is easy at every chunk size (paper@1: 32 to 36 out of 36). Finding the right passage is harder: the answer phrase was in the top k for only 15 to 27 of 36 questions. My first version of the test only checked the paper and scored 5/5 for every setting, so it could not tell the settings apart.
- No chunk size wins on everything. Size 800 has the best MRR (0.465) and the best paper@1 (36/36). Size 400 has the most answers in the top k (27/36) but the lowest MRR (0.364). Size 1200 has the fewest answers in the top k (15/36), although its MRR (0.428) is higher than that of 400.
- With 36 questions, a difference of one or two questions is noise, so this is a rough indication, not a benchmark.

**Bugs I found in my own test**

- One test phrase was wrong ("position encodings"; the paper writes "positional encoding"), so that question could never succeed.
- One phrase ("special classification token") was not found in the PDF text, probably because of a ligature character. Unicode normalization (NFKC) in the phrase check fixed it.

A sanity-check cell at the end of the notebook now verifies that every test phrase occurs in its PDF.

**Failure analysis**

Questions where the answer phrase was not in the top k at chunk size 800, read manually:

- Short factual answers lose to generic passages. "What does the abbreviation RAG stand for?" returned passages from the broader-impact section and the appendix instead of the abstract. The question shares few words with the sentence that answers it (my hypothesis, not tested).
- Wrong passage, sometimes wrong paper. "How long are the text passages stored in the document index?" returned RAG passages without the answer and one chunk from the BERT paper about pre-training.
- Formula noise. "How is retrieved text given to the generator together with the input?" returned chunks with garbled formula text from the PDF extraction (broken subscripts and symbols), which makes embeddings less reliable.

**Limitations**

- Text extraction from PDFs is imperfect for formulas and tables
- Fixed-size chunks can cut a sentence or an idea in the middle, and chunks are built page by page, so the overlap does not cross page boundaries
- k is rounded (3600 / 800 = 4.5 becomes 4), so the amount of context per chunk size varies between about 3200 and 3600 characters
- The chunk sizes were compared on the same 36 questions that are reported, with no separate held-out set, so the comparison is optimistic
- The phrase check ignores case, spaces and hyphens, and an answer phrase (especially a short number) can also appear in a chunk that does not answer the question
- The three papers are very different from each other, so finding the right paper is easy
- Test questions were drafted with AI assistance and checked against the PDFs

**What I would improve next**

- Add the generation step (an LLM answers from the retrieved chunks and cites them) and evaluate whether the answers stay faithful to the sources
- Hybrid search (keywords plus embeddings), because embeddings alone miss short factual sentences
- Chunk by sentences or sections instead of a fixed size, and avoid cutting across page boundaries
- Compare other embedding models and add a reranking step
- Use a larger test set with a separate held-out part

**What I learned**

- What is RAG, how it works, and what it is used for.
- A test can be too easy; checking only the paper gave near-perfect scores, checking the passage showed real differences.
- I found two mistakes in my own test set, so I now check the test before I trust the model.
- Perfectly balanced chunk size must not be too large (it will be noisy) or too small (model has a problem with understanding the context).

**Notes**

Parts of the code and the test questions were drafted with AI assistance. I ran all cells, debugged the issues described above, checked the test phrases against the PDFs and interpreted the results myself.
